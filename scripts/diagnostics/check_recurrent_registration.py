#!/usr/bin/env python3
"""Load the GRU and feed-forward task registrations inside Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result", default="logs/gates/recurrent_registration.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import uav_indoor.tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def fields(task: str) -> dict:
    cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    return {
        "task": task,
        "runner": type(cfg).__name__,
        "actor": type(cfg.actor).__name__,
        "critic": type(cfg.critic).__name__,
        "actor_hidden_dims": list(cfg.actor.hidden_dims),
        "critic_hidden_dims": list(cfg.critic.hidden_dims),
        "rnn_type": getattr(cfg.actor, "rnn_type", None),
        "rnn_hidden_dim": getattr(cfg.actor, "rnn_hidden_dim", None),
        "rnn_num_layers": getattr(cfg.actor, "rnn_num_layers", None),
        "num_steps_per_env": cfg.num_steps_per_env,
        "learning_rate": cfg.algorithm.learning_rate,
    }


try:
    recurrent = fields("Template-Uav-Building-Inspection-v0")
    feed_forward = fields("Template-Uav-Building-Inspection-FF-v0")
    passed = (
        recurrent["actor"] == "RslRlRNNModelCfg"
        and recurrent["rnn_type"] == "gru"
        and recurrent["rnn_hidden_dim"] == 64
        and recurrent["rnn_num_layers"] == 1
        and recurrent["actor_hidden_dims"] == [128, 64]
        and feed_forward["actor"] == "RslRlMLPModelCfg"
        and feed_forward["actor_hidden_dims"] == [128, 64]
        and recurrent["num_steps_per_env"] == feed_forward["num_steps_per_env"]
        and recurrent["learning_rate"] == feed_forward["learning_rate"]
    )
    result = {"verdict": "PASS" if passed else "FAIL", "recurrent": recurrent, "feed_forward": feed_forward}
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
finally:
    app.close()
