"""Fixed-capacity batched Kalman tracking for actual pose detections."""

from __future__ import annotations

import torch


class BatchedBoxKalman:
    """Constant-velocity Kalman filter over normalized ``cx, cy, w, h`` boxes."""

    def __init__(
        self,
        num_envs: int,
        max_tracks: int,
        device: torch.device | str,
        process_noise: float = 2.0e-3,
        # Variance in normalized image units. 1e-4 is a standard deviation of 0.01, or one
        # percent of image width, which is what a YOLO box actually jitters by. The old
        # 1e-2 claimed ten percent, so the filter distrusted the camera and lagged behind
        # real motion -- and the height servo reads this filtered box centre.
        measurement_noise: float = 1.0e-4,
        max_age_s: float = 0.75,
        association_distance: float = 0.25,
    ) -> None:
        self.num_envs = num_envs
        self.max_tracks = max_tracks
        self.device = torch.device(device)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_age_s = max_age_s
        self.association_distance = association_distance

        self.state = torch.zeros(num_envs, max_tracks, 8, device=self.device)
        eye = torch.eye(8, device=self.device)
        self.covariance = eye.expand(num_envs, max_tracks, 8, 8).clone()
        self.confidence = torch.zeros(num_envs, max_tracks, device=self.device)
        self.face_confidence = torch.zeros_like(self.confidence)
        self.age_s = torch.full_like(self.confidence, torch.inf)
        self.hits = torch.zeros(num_envs, max_tracks, dtype=torch.long, device=self.device)
        self.track_id = torch.full((num_envs, max_tracks), -1, dtype=torch.long, device=self.device)
        self._next_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self._identity = torch.eye(8, device=self.device)
        self._observation_matrix = torch.zeros(4, 8, device=self.device)
        self._observation_matrix[:, :4] = torch.eye(4, device=self.device)
        self._measurement_covariance = torch.eye(4, device=self.device) * measurement_noise
        self._env_index = torch.arange(num_envs, device=self.device)

    @property
    def valid(self) -> torch.Tensor:
        return (self.track_id >= 0) & (self.age_s <= self.max_age_s)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.state[env_ids] = 0.0
        self.covariance[env_ids] = torch.eye(8, device=self.device)
        self.confidence[env_ids] = 0.0
        self.face_confidence[env_ids] = 0.0
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
        process = torch.eye(8, device=self.device) * self.process_noise
        self.covariance = transition @ self.covariance @ transition.T + process
        self.age_s = torch.where(self.track_id >= 0, self.age_s + dt, self.age_s)
        expired = (self.track_id >= 0) & (self.age_s > self.max_age_s)
        self._drop(expired)

    def _drop(self, mask: torch.Tensor) -> None:
        self.track_id[mask] = -1
        self.confidence[mask] = 0.0
        self.face_confidence[mask] = 0.0
        self.hits[mask] = 0
        self.age_s[mask] = torch.inf

    def _correct(
        self,
        env_index: torch.Tensor,
        slot: torch.Tensor,
        measurement: torch.Tensor,
        confidence: torch.Tensor,
        face_confidence: torch.Tensor,
    ) -> None:
        """Batched Kalman correction for an arbitrary set of (env, slot) pairs."""

        h = self._observation_matrix
        covariance = self.covariance[env_index, slot]
        projected = h @ covariance
        innovation_covariance = projected @ h.T + self._measurement_covariance
        gain = torch.linalg.solve(innovation_covariance, projected).transpose(-1, -2)
        innovation = measurement - self.state[env_index, slot] @ h.T

        self.state[env_index, slot] += (gain @ innovation[..., None]).squeeze(-1)
        self.covariance[env_index, slot] = (self._identity - gain @ h) @ covariance
        self.confidence[env_index, slot] = confidence
        self.face_confidence[env_index, slot] = face_confidence
        self.age_s[env_index, slot] = 0.0
        self.hits[env_index, slot] += 1

    def update(
        self,
        boxes_xywhn: torch.Tensor,
        confidence: torch.Tensor,
        face_confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Associate and update a padded batch of actual detections.

        Greedy nearest-neighbour association, batched over environments: each round takes
        the single best pair per environment, so the loop is bounded by track count rather
        than environment count.
        """

        boxes_xywhn = boxes_xywhn.to(self.device)
        confidence = confidence.to(self.device)
        face_confidence = face_confidence.to(self.device)
        valid = valid.to(self.device)
        detections = boxes_xywhn.shape[1]

        active = self.track_id >= 0
        distance = torch.cdist(self.state[..., :2], boxes_xywhn[..., :2])
        distance = distance.masked_fill(~active[:, :, None] | ~valid[:, None, :], torch.inf)

        matched_detection = torch.full_like(self.track_id, -1)
        for _ in range(min(self.max_tracks, detections)):
            flat = distance.flatten(1)
            best = flat.argmin(dim=1)
            within = flat.gather(1, best[:, None]).squeeze(1) <= self.association_distance
            slot = torch.div(best, detections, rounding_mode="floor")
            detection = best % detections
            environments = self._env_index[within]
            matched_detection[environments, slot[within]] = detection[within]
            distance[environments, slot[within], :] = torch.inf
            distance[environments, :, detection[within]] = torch.inf

        env_index, slot = torch.nonzero(matched_detection >= 0, as_tuple=True)
        matched = torch.zeros_like(valid)
        if env_index.numel() > 0:
            detection = matched_detection[env_index, slot]
            matched[env_index, detection] = True
            self._correct(
                env_index,
                slot,
                boxes_xywhn[env_index, detection],
                confidence[env_index, detection],
                face_confidence[env_index, detection],
            )

        unmatched = valid & ~matched
        free = self.track_id < 0
        slot_rank = free.cumsum(dim=1) - 1
        detection_rank = unmatched.cumsum(dim=1) - 1
        pairing = (
            free[:, :, None]
            & unmatched[:, None, :]
            & (slot_rank[:, :, None] == detection_rank[:, None, :])
        )
        new_env, new_slot, new_detection = torch.nonzero(pairing, as_tuple=True)
        if new_env.numel() > 0:
            self.state[new_env, new_slot] = 0.0
            self.state[new_env, new_slot, :4] = boxes_xywhn[new_env, new_detection]
            self.covariance[new_env, new_slot] = self._identity * 0.1
            self.confidence[new_env, new_slot] = confidence[new_env, new_detection]
            self.face_confidence[new_env, new_slot] = face_confidence[new_env, new_detection]
            self.age_s[new_env, new_slot] = 0.0
            self.hits[new_env, new_slot] = 1
            self.track_id[new_env, new_slot] = self._next_id[new_env] + slot_rank[new_env, new_slot]
            self._next_id += pairing.any(dim=2).sum(dim=1)

    def step(
        self,
        dt: float,
        boxes_xywhn: torch.Tensor,
        confidence: torch.Tensor,
        face_confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        self.predict(dt)
        self.update(boxes_xywhn, confidence, face_confidence, valid)

    def observation(self) -> torch.Tensor:
        """Return fixed-size masked track features for the actor."""

        mask = self.valid.float()
        age = torch.where(self.valid, self.age_s, torch.zeros_like(self.age_s))
        features = torch.cat(
            [
                self.state,
                self.confidence[..., None],
                self.face_confidence[..., None],
                age[..., None],
                mask[..., None],
            ],
            dim=-1,
        )
        return features * mask[..., None]
