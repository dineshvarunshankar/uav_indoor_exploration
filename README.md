# UAV Indoor Exploration

A research prototype for Starling 2 UAV navigation and building-opening inspection in Isaac Lab.
It combines PPO policies, a PX4-style controller, relative-disparity perception, and a mission supervisor.

## Setup

Use a Python environment with Isaac Lab and RSL-RL installed.

```bash
git clone https://github.com/dineshvarunshankar/uav_indoor_exploration.git
cd uav_indoor_exploration
python -m pip install -e source/uav_indoor
```

Perception requires ONNX Runtime with CUDA support and Ultralytics.
Set the model and scene paths for your environment:

| Asset | Configuration |
|---|---|
| Starling 2 USD | [Robot configuration](source/uav_indoor/uav_indoor/assets/starling_2/starling_2.py) |
| Construction-site USD | [Building environment](source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/building_inspection/env_cfg.py) |
| MiDaS ONNX model | [MiDaS configuration](source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/perception/midas.py) |
| YOLOE weights and MobileCLIP prompt encoder | [Detector configuration](source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/perception/yoloe.py) |

The default USD paths require access to the AirLab Nucleus server. Model weights are external files.

## Run

List registered tasks:

```bash
python scripts/list_envs.py
```

Train obstacle navigation:

```bash
python scripts/rsl_rl/train.py \
  --task Template-Uav-Obstacle-Navigation-v0 \
  --num_envs 64 --headless --enable_cameras
```

Replay a checkpoint from the same task:

```bash
python scripts/rsl_rl/play.py \
  --task Template-Uav-Obstacle-Navigation-v0 \
  --checkpoint /path/to/model.pt \
  --num_envs 1 --enable_cameras
```

| Task | Environment ID |
|---|---|
| Indoor navigation | `Template-Uav-Indoor-v0` |
| Obstacle navigation | `Template-Uav-Obstacle-Navigation-v0` |
| Building inspection | `Template-Uav-Building-Inspection-v0` |
| Opening-entry curriculum | `Template-Uav-Entry-Curriculum-v0` |

## Project Structure

| Path | Contents |
|---|---|
| `source/uav_indoor/` | Robot assets, environments, policies, perception, and mission logic |
| `scripts/rsl_rl/` | Training, checkpoint replay, and curriculum evaluation |
| `scripts/diagnostics/` | Simulation diagnostics |
| `configs/` | Opening proposals and evaluation limits |
