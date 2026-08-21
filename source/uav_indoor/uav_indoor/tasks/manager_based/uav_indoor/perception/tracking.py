"""Short-term box tracking for prompted opening and person detections."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .prompts import DetectionRole
from .yoloe import DetectionBatch

TRACK_FEATURES = 11  # 8 state values, confidence, age, validity


@dataclass(frozen=True)
class TrackerCfg:
    """Filter and association constants for one role.

    PROVISIONAL and deliberately optimistic. Measure R from projected truth boxes, fit Q
    from truth box dynamics, then confirm with a normalized-innovation test.
    """

    max_tracks: int = 8
    process_noise: float = 6.0e-5
    measurement_noise: float = 4.0e-4
    max_age_s: float = 0.75
    association_distance: float = 0.20

    @classmethod
    def for_openings(cls) -> TrackerCfg:
        """An opening does not move. Its box moves only because the aircraft moves.

        At a scan station the aircraft holds position, so the box is almost static and the
        process noise covers attitude jitter alone. Openings are sparse on a facade, so a
        wide gate is safe. A long life keeps a candidate through several missed frames.
        """

        return cls(
            process_noise=1.0e-5,
            measurement_noise=2.0e-4,
            max_age_s=1.5,
            association_distance=0.25,
        )

    @classmethod
    def for_people(cls) -> TrackerCfg:
        """A person walks, so the process noise covers their acceleration.

        People stand close together, so the gate stays tighter than for openings. An
        identity swap would restart the 10-second observation hold.
        """

        return cls()


class BatchedPromptBoxKalman:
    """Constant-velocity Kalman filter over normalized ``cx, cy, w, h`` boxes.

    One instance holds one mission role, so opening and person identifiers stay
    separate and no detection can associate across roles.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: TrackerCfg | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or TrackerCfg()
        max_tracks = self.cfg.max_tracks
        self.max_tracks = max_tracks

        self.state = torch.zeros(num_envs, max_tracks, 8, device=self.device)
        eye = torch.eye(8, device=self.device)
        self.covariance = eye.expand(num_envs, max_tracks, 8, 8).clone()
        self.confidence = torch.zeros(num_envs, max_tracks, device=self.device)
        self.age_s = torch.full_like(self.confidence, torch.inf)
        self.hits = torch.zeros(num_envs, max_tracks, dtype=torch.long, device=self.device)
        self.track_id = torch.full((num_envs, max_tracks), -1, dtype=torch.long, device=self.device)
        self._next_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self._identity = eye
        self._observation_matrix = torch.zeros(4, 8, device=self.device)
        self._observation_matrix[:, :4] = torch.eye(4, device=self.device)
        self._measurement_covariance = torch.eye(4, device=self.device) * self.cfg.measurement_noise
        self._env_index = torch.arange(num_envs, device=self.device)

    @property
    def valid(self) -> torch.Tensor:
        return (self.track_id >= 0) & (self.age_s <= self.cfg.max_age_s)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.state[env_ids] = 0.0
        self.covariance[env_ids] = self._identity
        self.confidence[env_ids] = 0.0
        self.age_s[env_ids] = torch.inf
        self.hits[env_ids] = 0
        self.track_id[env_ids] = -1
        self._next_id[env_ids] = 0

    def predict(self, dt: float) -> None:
        transition = torch.eye(8, device=self.device)
        transition[0, 4] = dt
        transition[1, 5] = dt
        transition[2, 6] = dt
        transition[3, 7] = dt
        self.state = torch.einsum("ij,etj->eti", transition, self.state)
        process = torch.eye(8, device=self.device) * self.cfg.process_noise
        self.covariance = transition @ self.covariance @ transition.T + process
        self.age_s = torch.where(self.track_id >= 0, self.age_s + dt, self.age_s)
        self._drop((self.track_id >= 0) & (self.age_s > self.cfg.max_age_s))

    def _drop(self, mask: torch.Tensor) -> None:
        self.track_id = torch.where(mask, torch.full_like(self.track_id, -1), self.track_id)
        self.confidence = torch.where(mask, torch.zeros_like(self.confidence), self.confidence)
        self.hits = torch.where(mask, torch.zeros_like(self.hits), self.hits)
        self.age_s = torch.where(mask, torch.full_like(self.age_s, torch.inf), self.age_s)

    def _associate(
        self,
        boxes_xywhn: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy nearest-neighbour association. Returns a detection index per slot, or -1.

        Every round takes the best remaining pair in each environment at the same time.
        The round count is bounded by the track count, never by the environment count, and
        no operation reads a value back to the host.
        """

        detections = boxes_xywhn.shape[1]
        slot_range = torch.arange(self.max_tracks, device=self.device)
        detection_range = torch.arange(detections, device=self.device)

        active = self.track_id >= 0
        distance = torch.cdist(self.state[..., :2], boxes_xywhn[..., :2])
        distance = distance.masked_fill(~active[:, :, None] | ~valid[:, None, :], torch.inf)

        matched_detection = torch.full_like(self.track_id, -1)
        for _ in range(min(self.max_tracks, detections)):
            flat = distance.flatten(1)
            best = flat.argmin(dim=1)
            within = flat.gather(1, best[:, None]).squeeze(1) <= self.cfg.association_distance
            slot = torch.div(best, detections, rounding_mode="floor")
            detection = best % detections
            slot_hit = (slot_range[None, :] == slot[:, None]) & within[:, None]
            detection_hit = (detection_range[None, :] == detection[:, None]) & within[:, None]
            chosen = detection[:, None].expand_as(matched_detection)
            matched_detection = torch.where(slot_hit, chosen, matched_detection)
            distance = distance.masked_fill(slot_hit[:, :, None] | detection_hit[:, None, :], torch.inf)
        return matched_detection

    def _correct(
        self,
        matched: torch.Tensor,
        measurement: torch.Tensor,
        confidence: torch.Tensor,
    ) -> None:
        """Kalman correction over the whole (env, slot) grid, applied where matched."""

        h = self._observation_matrix
        projected = h @ self.covariance
        innovation_covariance = projected @ h.T + self._measurement_covariance
        gain = torch.linalg.solve(innovation_covariance, projected).transpose(-1, -2)
        innovation = measurement - self.state @ h.T

        state = self.state + (gain @ innovation[..., None]).squeeze(-1)
        covariance = (self._identity - gain @ h) @ self.covariance
        self.state = torch.where(matched[..., None], state, self.state)
        self.covariance = torch.where(matched[..., None, None], covariance, self.covariance)
        self.confidence = torch.where(matched, confidence, self.confidence)
        self.age_s = torch.where(matched, torch.zeros_like(self.age_s), self.age_s)
        self.hits = torch.where(matched, self.hits + 1, self.hits)

    def update(
        self,
        boxes_xywhn: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Associate and update a padded batch of detections of this role."""

        boxes_xywhn = boxes_xywhn.to(self.device)
        confidence = confidence.to(self.device)
        valid = valid.to(self.device)

        matched_detection = self._associate(boxes_xywhn, valid)
        matched = matched_detection >= 0
        index = matched_detection.clamp_min(0)
        self._correct(
            matched,
            boxes_xywhn.gather(1, index[..., None].expand(-1, -1, 4)),
            confidence.gather(1, index),
        )

        # Free slots take unmatched detections in order. A full role drops the rest.
        taken = torch.zeros_like(valid, dtype=torch.long)
        taken.scatter_reduce_(1, index, matched.long(), reduce="amax", include_self=True)
        unmatched = valid & ~taken.bool()
        free = self.track_id < 0
        slot_rank = free.cumsum(dim=1) - 1
        detection_rank = unmatched.cumsum(dim=1) - 1
        pairing = (
            free[:, :, None]
            & unmatched[:, None, :]
            & (slot_rank[:, :, None] == detection_rank[:, None, :])
        )
        started = pairing.any(dim=2)
        source = pairing.long().argmax(dim=2)
        box = boxes_xywhn.gather(1, source[..., None].expand(-1, -1, 4))
        state = torch.cat((box, torch.zeros_like(box)), dim=-1)
        self.state = torch.where(started[..., None], state, self.state)
        self.covariance = torch.where(started[..., None, None], self._identity * 0.1, self.covariance)
        self.confidence = torch.where(started, confidence.gather(1, source), self.confidence)
        self.age_s = torch.where(started, torch.zeros_like(self.age_s), self.age_s)
        self.hits = torch.where(started, torch.ones_like(self.hits), self.hits)
        self.track_id = torch.where(started, self._next_id[:, None] + slot_rank, self.track_id)
        self._next_id = self._next_id + started.sum(dim=1)

    def step(
        self,
        dt: float,
        boxes_xywhn: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        self.predict(dt)
        self.update(boxes_xywhn, confidence, valid)

    def observation(self) -> torch.Tensor:
        """Return fixed-size masked track features for the actor."""

        mask = self.valid.float()
        age = torch.where(self.valid, self.age_s, torch.zeros_like(self.age_s))
        features = torch.cat(
            [
                self.state,
                self.confidence[..., None],
                age[..., None],
                mask[..., None],
            ],
            dim=-1,
        )
        return features * mask[..., None]


class PromptBoxTrackers:
    """One tracker per mission role, advanced from one detection batch."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        opening_cfg: TrackerCfg | None = None,
        person_cfg: TrackerCfg | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.openings = BatchedPromptBoxKalman(num_envs, device, opening_cfg or TrackerCfg.for_openings())
        self.people = BatchedPromptBoxKalman(num_envs, device, person_cfg or TrackerCfg.for_people())

    def tracker_for_role(self, role: DetectionRole) -> BatchedPromptBoxKalman:
        return self.openings if role is DetectionRole.OPENING_CANDIDATE else self.people

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.openings.reset(env_ids)
        self.people.reset(env_ids)

    def predict(self, dt: float) -> None:
        self.openings.predict(dt)
        self.people.predict(dt)

    def update(self, batch: DetectionBatch) -> None:
        """Route one detection batch to both role trackers by validity mask."""

        for role in (DetectionRole.OPENING_CANDIDATE, DetectionRole.PERSON):
            self.tracker_for_role(role).update(
                batch.boxes_xywhn,
                batch.confidence,
                batch.valid_for_role(role),
            )

    def step(self, dt: float, batch: DetectionBatch) -> None:
        self.predict(dt)
        self.update(batch)

    def observation(self) -> torch.Tensor:
        """Return both role blocks as one flat actor feature vector."""

        return torch.cat((self.openings.observation().flatten(1), self.people.observation().flatten(1)), dim=-1)
