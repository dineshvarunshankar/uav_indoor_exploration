from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from rlPx4Controller.pyParallelControl import ParallelVelControl

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

@configclass
class PX4VelocityActionCfg(ActionTermCfg):
    """linear velocity + yaw rate setpoint action via rlPx4 ParallelVelControl."""

    class_type:type[ActionTerm] = None
    joint_names: list[str] = ["joint0", "joint1", "joint2", "joint3"]

    v_max_xy: float = 3.0
    v_max_z : float = 1.0
    yaw_rate_max: float = 2.618 # 150 deg/s
    motor_scale: float = 1.0
    motor_clip: tuple[float, float] | None = None


class PX4VelocityAction(ActionTerm):
    cfg: PX4VelocityActionCfg

    def __init__(self, cfg: PX4VelocityActionCfg, env: ManagerBasedEnv)-> None:
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._motor_cmds = torch.zeros(self.num_envs, 4, device=self.device)

        self._scale = torch.tensor(
            [cfg.v_max_xy, cfg.v_max_xy, cfg.v_max_z, cfg.yaw_rate_max],
            device=self.device,
        )

        #rlpx4 is cpu/numpy
        self._px4 = ParallelVelControl(self.num_envs)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        "once per env step: convert policy action to velocity + yaw rate setpoints for rlPx4"
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions * self._scale

    def apply_actions(self) -> None:
        "once per  physics step: state -> rlpx4 ->rotor effort commands"

        data = self._asset.data
        dt = self._env.physics_dt

        vel_b = self._processed_actions[:, :3]
        vel_w = math.utils.quat_apply(quat, vel_b)
        actions_w = torch.cat([vel_w, self._processed_actions[:, 3:4]], dim=-1)
        actions_np = actions_w.detach().cpu().numpy()

        pos = data.root_pos_w.detach().cpu().numpy()
        vel = data.root_lin_vel_w.detach().cpu().numpy()
        ang_vel = data.root_ang_vel_w.detach().cpu().numpy()
        quat = data.root_quat_w.detach().cpu().numpy()
        
        self._px4.set_status(pos, quat, vel, ang_vel, dt)
        motor_np = self._px4.update(actions_np)

        self._motor_cmds[:] = torch.as_tensor(motor_np, device=self.device, dtype=dtype.float32)
        self._motor_cmds *= self.cfg.motor_scale

        if self.cfg.motor_clip is not None:
            lo, hi = self.cfg.motor_clip
            self._motor_cmds.clamp_(lo, hi)
        
        self._asset.set_joint_effort_target(self._motor_cmds, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._motor_cmds.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._motor_cmds[env_ids] = 0.0

PX4VelocityActionCfg.class_type = PX4VelocityAction