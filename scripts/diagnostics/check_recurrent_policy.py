#!/usr/bin/env python3
"""Qualify the deployable 96-to-4 GRU actor without launching Isaac Sim."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models.rnn_model import RNNModel


def build(batch: int) -> RNNModel:
    obs = TensorDict({"policy": torch.zeros(batch, 96)}, batch_size=[batch])
    return RNNModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=4,
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
    ).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default="logs/gates/recurrent_policy.json")
    parser.add_argument("--timing_steps", type=int, default=1000)
    args = parser.parse_args()
    torch.manual_seed(7)
    model = build(batch=2)
    zero = TensorDict({"policy": torch.zeros(2, 96)}, batch_size=[2])
    prior = TensorDict({"policy": torch.ones(2, 96)}, batch_size=[2])

    with torch.inference_mode():
        model(prior)
        hidden_before = model.get_hidden_state().clone()
        model.reset(torch.tensor([1, 0]))
        hidden_after = model.get_hidden_state().clone()
        selective_reset = bool(
            torch.count_nonzero(hidden_after[:, 0]).item() == 0
            and torch.equal(hidden_after[:, 1], hidden_before[:, 1])
        )

        model.reset()
        expected = model(zero).clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = build(batch=2)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            restored.reset()
            checkpoint_match = bool(torch.allclose(expected, restored(zero), atol=1.0e-6, rtol=0.0))

            model.reset()
            exported = torch.jit.script(model.as_jit())
            export_path = Path(directory) / "actor_jit.pt"
            exported.save(str(export_path))
            reloaded = torch.jit.load(str(export_path))
            export_output = reloaded(torch.zeros(1, 96))
            export_shape = list(export_output.shape)

        timing_model = torch.jit.script(build(batch=1).as_jit())
        sample = torch.zeros(1, 96)
        for _ in range(20):
            timing_model(sample)
        started = time.perf_counter()
        for _ in range(args.timing_steps):
            timing_model(sample)
        latency_ms = 1000.0 * (time.perf_counter() - started) / args.timing_steps

    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "verdict": "PASS" if selective_reset and checkpoint_match and export_shape == [1, 4] else "FAIL",
        "input_dim": 96,
        "output_dim": 4,
        "rnn_type": "gru",
        "rnn_hidden_dim": 64,
        "rnn_num_layers": 1,
        "hidden_dims": [128, 64],
        "selective_environment_reset": selective_reset,
        "checkpoint_reload_match": checkpoint_match,
        "torchscript_output_shape": export_shape,
        "parameters": parameters,
        "parameter_bytes_float32": parameters * 4,
        "cpu_inference_latency_ms": latency_ms,
        "timing_steps": args.timing_steps,
    }
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
