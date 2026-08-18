# Animated Crowd Design for Casualty Inspection

Status: Approved design

Date: 2026-08-18

## Purpose

The casualty-inspection task needs ten animated people in each parallel environment. Four people are inspection targets, and six people are distractors.

All people remain upright and use skeletal animation. They continue their behavior across drone and mission resets.

## Problem

The current task moves one person by writing a root transform. This method causes visible sliding because it does not control a walking animation.

The reset event also assigns new random positions to all casualty models. Curriculum stage 0 ends after approximately three steps, so characters teleport frequently.

The current posture table contains large roll rotations. Some character models consequently face the ground or rotate around an unsuitable model pivot.

## Goals

- Spawn ten fully animated people in every replicated environment.
- Give each person bounded and diverse behavior.
- Keep all root transforms upright.
- Keep crowd motion continuous across drone resets.
- Let selected targets walk during inspection.
- Keep collision proxies synchronized with visible characters.
- Add poles, boxes, and barriers as scene obstacles.
- Keep the crowd code separate from the mission and perception code.
- Preserve deterministic training through per-environment random seeds.

## Non-goals

- The task does not import the Pegasus `Person` or `PeopleManager` classes at runtime.
- The task does not use the complete Isaac Replicator Agent simulation lifecycle.
- The first version does not implement a social-force crowd model.
- The first version does not support sitting, queues, or character interaction animations.

## Selected Approach

The task uses a task-native crowd manager. It follows the animation-graph control pattern used by Pegasus Simulator.

The manager integrates with Isaac Lab environment namespaces and vectorized state. It uses Isaac animation APIs directly.

This approach preserves independent environment resets. It also avoids the global `World` and `/World/Characters` assumptions in Pegasus.

## Code Structure

The task adds one `crowd` package with three modules:

- `config.py` contains assets, movement limits, route limits, and behavior probabilities.
- `manager.py` contains vectorized crowd state and the behavior state machine.
- `animation.py` connects desired motion to Isaac animation graphs.

The environment creates one `CrowdManager`. The environment only updates the manager and reads its public state.

The environment does not contain animation-graph commands or pedestrian behavior rules.

## Crowd State

The manager stores each field with shape `[num_envs, 10, ...]` where applicable:

- World position.
- Horizontal velocity.
- Heading.
- Current waypoint.
- Current behavior.
- Behavior time remaining.
- Character asset index.
- Target identity.
- Active inspection identity.

Four fixed indices are targets in each environment. Six fixed indices are distractors.

The mission receives target truth only through privileged state. The actor continues to use camera-derived pose tracks.

## Character Spawning and Animation

Each environment contains ten character prims under its environment namespace. Each character uses a supported Isaac Sim person asset.

The animation adapter attaches and validates the character animation graph. It selects `Walk`, `Idle`, and look-around states.

A horizontal velocity more than `0.1 m/s` selects walking. A lower velocity selects an idle state.

The heading follows horizontal velocity. The root roll and pitch remain zero.

The adapter does not silently use transform-only sliding. Missing assets, skeletons, or animation graphs cause a startup error.

## Behaviors

Each person uses one finite-state behavior machine. The machine contains these states:

- `PATROL`: Move through a fixed waypoint loop.
- `WANDER`: Select a reachable random waypoint inside the movement area.
- `IDLE`: Stop for a bounded duration.
- `LOOK_AROUND`: Play an idle look animation without changing position.

The default state probabilities are 40 percent patrol, 30 percent wander, 20 percent idle, and 10 percent look-around.

Walking speed is between `0.5 m/s` and `1.2 m/s`. An inspected target uses a maximum speed of `1.0 m/s`.

Behavior transitions occur after 3 to 12 seconds. An inspected target does not make an abrupt route reversal during active recording.

Routes stay within 18 meters of each environment origin along both horizontal axes. Route generation rejects points inside obstacle clearance zones.

## Moving-target Inspection

The selected target continues walking during approach, face acquisition, recording, and the inspection arc.

The mission reference uses the current target position and filtered velocity. It predicts a short target lead position for drone tracking.

The inspection arc is relative to the moving target. Its center updates continuously without resetting arc progress.

Recording pauses during invalid face visibility or occlusion. Recording resumes when validity returns.

## Collision and Obstacles

Each visible person has one kinematic capsule proxy. The proxy follows the pelvis position and horizontal heading on every environment step.

The scene adds simple poles, boxes, and barriers. These objects use Isaac primitive shapes and collision properties.

The route generator prevents planned paths through static obstacles. Lightweight local steering prevents direct person-to-person overlap.

The privileged clearance state includes people and artificial obstacles. The actor does not receive their exact positions.

## Reset Rules

A drone episode reset changes the drone, mission state, observations, and rewards. It does not change crowd positions or behavior states.

A crowd reset occurs only at scene startup or through an explicit full-scene reset. This reset uses a deterministic per-environment seed.

The design removes random casualty repositioning from the normal mission reset. It also removes arbitrary casualty posture rotations.

## Curriculum

The task uses three curriculum stages:

1. Ten people follow predictable patrol routes. The policy learns flight and moving-target tracking.
2. People use all behavior states. The scene adds artificial obstacles and distractors.
3. The task enables the complete multi-target inspection mission.

The curriculum changes mission difficulty only. It does not teleport or recreate people.

Automatic stage changes remain optional. A training command can select a fixed stage.

## Update Order

Each environment step uses this order:

1. The crowd manager updates behaviors and desired motion.
2. The animation adapter updates animation state and heading.
3. The manager synchronizes collision proxies and privileged state.
4. Isaac Sim renders the camera image.
5. MiDaS and pose estimation update perception tracks.
6. The mission supervisor updates the moving mission reference.
7. Isaac Lab applies the drone action and computes task results.

## Error Handling and Metrics

Startup errors identify the missing asset, skeleton, animation graph, or extension. The task stops before training begins.

Runtime validation rejects non-finite positions, invalid behavior values, and positions outside configured bounds.

Metrics include crowd speed, behavior distribution, proxy error, target continuity, animation state, and crowd update latency.

## Tests

Pure Torch tests cover behavior transitions, speed limits, headings, boundaries, deterministic seeds, and reset continuity.

Isaac integration tests cover character spawning, upright transforms, animation graphs, animation states, and proxy synchronization.

Mission tests cover a walking target, continuous arc progress, temporary occlusion, and recording recovery.

Smoke tests use at least two parallel environments with ten animated people each. A second smoke test uses four environments.

The PPO smoke test must complete one iteration without a traceback, invalid tensor, missing asset, or GPU memory error.

## Acceptance Criteria

- Each environment contains ten visible and animated people.
- Four people are targets, and six people are distractors.
- Walking people use walking animation and face their direction of travel.
- Idle people use an idle animation.
- Character root roll and pitch remain within five degrees.
- A drone reset does not cause a crowd position discontinuity.
- Selected targets remain in motion during inspection.
- Collision proxies remain within `0.10 m` of their visible characters.
- People and planned routes remain inside configured environment bounds.
- Two-environment and four-environment smoke tests complete successfully.
- The task has no runtime dependency on Pegasus Simulator.

## References

- [Pegasus simulation with moving people](https://pegasussimulator.github.io/PegasusSimulator/source/tutorials/create_simulation_with_people.html)
- [Isaac Sim 5.1 actor control](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/action_and_event_data_generation/ext_replicator-agent/actor_control.html)
