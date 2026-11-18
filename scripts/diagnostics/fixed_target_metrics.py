"""Per-aircraft qualification of a fixed position and heading target."""

import numpy as np

LIMITS = {
    "position_peak_m": 0.3,
    "heading_peak_rad": float(np.deg2rad(5)),
    "tilt_peak_rad": float(np.deg2rad(10)),
    "body_rate_vector_rms_rad_s": 0.15,
    "whole_run_position_peak_m": 3.0,
    "whole_run_tilt_peak_rad": float(np.deg2rad(45)),
}

def qualify(data, physics_dt, settle_s=4.0):
    invalid = []
    required = ("time", "position", "target", "angles", "heading", "rates")
    if any(key not in data for key in required):
        return {"verdict": "INVALID", "invalid": ["missing trace field"], "per_env_pass": []}
    data = {key: np.asarray(data[key]) for key in required}
    time = data["time"]
    if not np.isfinite(physics_dt) or physics_dt <= 0 or not np.isfinite(settle_s) or settle_s < 0:
        invalid.append("invalid timing configuration")
    if any(not np.isfinite(value).all() for value in data.values()):
        invalid.append("non-finite sample")
    if time.ndim != 1 or len(time) < 2:
        return {"verdict": "INVALID", "invalid": invalid + ["insufficient timestamps"], "per_env_pass": []}
    position = data["position"]
    if position.ndim != 3 or position.shape[-1] != 3 or position.shape[1] < 1:
        return {"verdict": "INVALID", "invalid": invalid + ["invalid position shape"], "per_env_pass": []}
    count, envs, _ = position.shape
    shapes = {"time": (count,), "position": (count, envs, 3), "target": (count, envs, 3),
              "angles": (count, envs, 3), "heading": (count, envs), "rates": (count, envs, 3)}
    if any(data[key].shape != shape for key, shape in shapes.items()):
        return {"verdict": "INVALID", "invalid": invalid + ["inconsistent trace shapes"], "per_env_pass": []}
    if not np.allclose(np.diff(time), physics_dt, atol=1e-6, rtol=1e-4):
        invalid.append("samples are not at physics rate")
    if not np.isclose(time[0], physics_dt, atol=1e-6, rtol=1e-4):
        invalid.append("trace does not start at the first physics step")
    settled = time >= settle_s
    observed_duration = np.count_nonzero(settled) * physics_dt
    if np.count_nonzero(settled) < 2 or observed_duration < 2 - 1e-6:
        invalid.append("less than two seconds after settling")
    if not np.allclose(data["target"], data["target"][0], atol=1e-6, rtol=0):
        invalid.append("position target moved")
    if not np.allclose(data["heading"], data["heading"][0], atol=1e-6, rtol=0):
        invalid.append("heading target moved")
    if invalid:
        return {"verdict": "INVALID", "invalid": invalid, "per_env_pass": [False] * envs}
    error = np.linalg.norm(position - data["target"], axis=-1)
    yaw_error = data["heading"] - data["angles"][:, :, 2]
    yaw_error = np.abs(np.arctan2(np.sin(yaw_error), np.cos(yaw_error)))
    tilt = np.linalg.norm(data["angles"][:, :, :2], axis=-1)
    rates_rms = np.sqrt(np.mean(np.sum(data["rates"][settled] ** 2, axis=-1), axis=0))
    metrics = {
        "position_peak_m": error[settled].max(axis=0),
        "heading_peak_rad": yaw_error[settled].max(axis=0),
        "tilt_peak_rad": tilt[settled].max(axis=0),
        "body_rate_vector_rms_rad_s": rates_rms,
        "whole_run_position_peak_m": error.max(axis=0),
        "whole_run_tilt_peak_rad": tilt.max(axis=0),
    }
    failed = [[key for key, values in metrics.items() if values[i] > LIMITS[key]] for i in range(envs)]
    passes = [not failures for failures in failed]
    return {"verdict": "PASS" if all(passes) else "FAIL", "invalid": [], "limits": LIMITS,
            "per_env_pass": passes, "per_env_failures": failed,
            "per_env_metrics": {key: value.tolist() for key, value in metrics.items()},
            "per_env_body_rate_vector_rms": rates_rms.tolist()}
