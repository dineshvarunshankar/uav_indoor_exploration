#!/usr/bin/env python3
"""Bounded paired perception diagnostic; writes progress before each subprocess."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path('/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python')
PARENT = ROOT / 'logs/rsl_rl/uav_t6_smoke/2026-09-13_17-11-19_representative_lazy_gamma099_lam095/model_80.pt'
OUT = ROOT / 'logs/gates/perception_comparison_20260913'
SOURCES = ('midas', 'rendered_inverse_depth')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'report.json').exists():
        raise RuntimeError('Comparison already exists; do not overwrite its evidence')
    report = {'status': 'RUNNING', 'parent': str(PARENT),
              'parent_sha256': hashlib.sha256(PARENT.read_bytes()).hexdigest(),
              'training_seed_start': 2600000, 'evaluation_seed_start': 3600000,
              'updates_per_source': 30, 'transitions_per_source': 64 * 256 * 30,
              'normalization': 'per-frame 2/98 percentile inverse depth; no metric scale',
              'runs': []}

    files = [ROOT / 'scripts/rsl_rl/evaluate_obstacle_navigation.py']
    files += list((ROOT / 'source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor').rglob('*.py'))
    report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

    def save():
        temporary = OUT / 'report.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(OUT / 'report.json')

    def run(command, label):
        report['active'] = label
        save()
        with (OUT / f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    def train(source, updates, label):
        experiment = 'uav_perception_comparison'
        log_root = ROOT / 'logs/rsl_rl' / experiment
        before = set(log_root.glob('*'))
        command = [str(PYTHON), 'scripts/rsl_rl/train.py', '--task',
                   'Template-Uav-Obstacle-Navigation-v0', '--headless', '--enable_cameras',
                   '--device', 'cuda:0', '--num_envs', '64', '--max_iterations', str(updates),
                   '--save_interval', str(updates), '--experiment_name', experiment,
                   '--run_name', label, '--resume', '--resume_path', str(PARENT),
                   'env.initial_difficulty=1', 'env.training_seed_start=2600000',
                   'env.layout_bank_size=256', 'env.obstacle_count_overrides=[0,1,12,20]',
                   'agent.num_steps_per_env=256', f'env.disparity.source={source}']
        run(command, label)
        created = set(log_root.glob('*')) - before
        if len(created) != 1:
            raise RuntimeError(f'Expected one new run, found {created}')
        directory = created.pop()
        checkpoint = max(directory.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
        report['runs'].append({'source': source, 'updates': updates, 'checkpoint': str(checkpoint)})
        save()
        return checkpoint

    try:
        for source in SOURCES:
            train(source, 1, f'smoke_{source}')
        for source in SOURCES:
            checkpoint = train(source, 30, f'paired30_{source}')
            for difficulty in (0, 1):
                label = f'evaluate_{source}_{difficulty}'
                run([str(PYTHON), 'scripts/rsl_rl/evaluate_obstacle_navigation.py',
                     '--checkpoint', str(checkpoint), '--difficulty', str(difficulty),
                     '--episodes', '32', '--num_envs', '32', '--seed-start', '3600000',
                     '--obstacle-count', str(difficulty), '--image-source', source,
                     '--output', str(OUT / f'{label}.json'), '--headless', '--device', 'cuda:0'], label)
        report['status'] = 'RECORDED'
        report.pop('active', None)
    except Exception as error:
        report['status'] = 'ERROR'
        report['error'] = repr(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
