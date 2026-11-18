"""Thin Isaac animation-graph adapter for a vectorized crowd."""

from __future__ import annotations

import numpy as np
import torch

import carb
import omni.anim.graph.core as ag
import omni.usd
from isaacsim.replicator.agent.core.stage_util import CharacterUtil
from pxr import Gf, UsdGeom


class AnimatedCrowdAdapter:
    """Attach and command animated characters inside Isaac Lab namespaces."""

    def __init__(
        self,
        num_envs: int,
        num_people: int,
        num_casualties: int,
        device: str | torch.device = "cpu",
        graph_root_path: str = "/World/Characters/Biped_Setup",
    ) -> None:
        self.num_envs = num_envs
        self.num_people = num_people
        self.num_casualties = num_casualties
        self.device = torch.device(device)
        self.graph_root_path = graph_root_path
        self._walker_slots: list[tuple[int, int]] = []
        self._walker_paths: list[str] = []
        self._characters: list[object | None] = []
        self._casualty_prims: list[tuple[int, int, object]] = []
        self._casualty_positions = torch.zeros(num_envs, num_casualties, 3, device=self.device)

    def attach(self) -> None:
        """Bind the shared graph to the walkers and collect the casualty prims."""
        stage = omni.usd.get_context().get_stage()
        graph_root = stage.GetPrimAtPath(self.graph_root_path)
        graph = CharacterUtil.get_anim_graph_from_character(graph_root)
        if graph is None or not graph.IsValid():
            raise RuntimeError(f"Animation graph is unavailable at {self.graph_root_path}")

        skeletons = []
        self._walker_slots, self._walker_paths, self._casualty_prims = [], [], []
        for env_index in range(self.num_envs):
            for person_index in range(self.num_people):
                root_path = f"/World/envs/env_{env_index}/Person{person_index}"
                root = stage.GetPrimAtPath(root_path)
                if not root.IsValid():
                    raise RuntimeError(f"Character prim is unavailable at {root_path}")
                if person_index < self.num_casualties:
                    self._casualty_prims.append((env_index, person_index, root))
                    continue
                skeleton = CharacterUtil.get_character_skelroot_by_root(root)
                if skeleton is None or not skeleton.IsValid():
                    raise RuntimeError(f"Character skeleton is unavailable at {root_path}")
                skeletons.append(skeleton)
                self._walker_slots.append((env_index, person_index))
                self._walker_paths.append(str(skeleton.GetPath()))

        CharacterUtil.setup_animation_graph_to_character(skeletons, graph)
        self._characters = [None] * len(self._walker_paths)

    def place_casualties(
        self,
        positions_w: torch.Tensor,
        orientations: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Write casualty pose straight onto the prim."""

        expected = (self.num_envs, self.num_casualties)
        if positions_w.shape != (*expected, 3) or orientations.shape != (*expected, 4):
            raise ValueError(f"expected positions {(*expected, 3)} and orientations {(*expected, 4)}")

        self._casualty_positions = positions_w.to(self.device).clone()
        host_position = (positions_w - env_origins[:, None, :]).detach().cpu().numpy()
        host_orientation = orientations.detach().cpu().numpy()
        for env_index, person_index, prim in self._casualty_prims:
            position = host_position[env_index, person_index]
            w, x, y, z = host_orientation[env_index, person_index]
            matrix = Gf.Matrix4d()
            matrix.SetRotate(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
            matrix.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)

    def _ensure_handles(self) -> None:
        for index, path in enumerate(self._walker_paths):
            if self._characters[index] is None:
                self._characters[index] = ag.get_character(path)
            if self._characters[index] is None:
                raise RuntimeError(f"Animation runtime handle is unavailable for {path}")

    def command(
        self,
        destination: torch.Tensor,
        speed: torch.Tensor,
        moving: torch.Tensor,
    ) -> None:
        """Send walk or idle commands to the walkers."""
        expected = (self.num_envs, self.num_people)
        if destination.shape != (*expected, 3):
            raise ValueError(f"destination must have shape {(*expected, 3)}")
        if speed.shape != expected or moving.shape != expected:
            raise ValueError(f"speed and moving must have shape {expected}")
        self._ensure_handles()

        host_destination = destination.detach().cpu()
        host_speed = speed.detach().cpu()
        host_moving = moving.detach().cpu()
        for index, character in enumerate(self._characters):
            env_index, person_index = self._walker_slots[index]
            if host_moving[env_index, person_index]:
                start = carb.Float3()
                rotation = carb.Float4()
                character.get_world_transform(start, rotation)
                goal = carb.Float3(*host_destination[env_index, person_index].tolist())
                character.set_variable("Action", "Walk")
                character.set_variable("PathPoints", [start, goal])
                character.set_variable("Walk", float(host_speed[env_index, person_index]))
            else:
                character.set_variable("Walk", 0.0)
                character.set_variable("Action", "Idle")

    def positions(self) -> torch.Tensor:
        """Return every person's position: walkers from the runtime, casualties as placed."""
        self._ensure_handles()
        output = np.empty((len(self._characters), 3), dtype=np.float32)
        position = carb.Float3()
        rotation = carb.Float4()
        for index, character in enumerate(self._characters):
            character.get_world_transform(position, rotation)
            output[index] = (position.x, position.y, position.z)

        result = torch.zeros(self.num_envs, self.num_people, 3, device=self.device)
        result[:, : self.num_casualties] = self._casualty_positions
        walkers = torch.from_numpy(output).to(self.device)
        for index, (env_index, person_index) in enumerate(self._walker_slots):
            result[env_index, person_index] = walkers[index]
        return result
