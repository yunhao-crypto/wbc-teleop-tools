# SONIC ONNX planner and map bridge

This optional chain runs `planner_sonic.onnx` and `bridge_map.onnx` with Python
ONNX Runtime. It combines planner legs and root with Pico arms before passing a
motion reference to the existing SONIC actor. It does not load an HBM model.
The feature is selected only with `--planner-config`; the ordinary Pico path is
unchanged. The included configuration uses the 9700-step actor from
`config/actor_pico.json`, the new map bridge, 0.5 m/s forward/backward and
0.5 rad/s turn commands. These speeds are experimental. S100 offline installation
and synthetic model checks are recorded in the [delivery guide](README.md#s100-offline-validation).
Live tracking and motion acceptance remain unmeasured.

## External models

All three ONNX weights stay outside Git. Their installation paths, byte counts
and SHA256 identities are pinned in `config/actor_pico.json` and
`config/actor_sonic_planner.json`; the delivery manifest records them too.
First build with `python3 scripts/build_actor_delivery.py --variant planner`,
then copy and extract that complete package on S100. Its requirements include
Python ONNX Runtime; the base package excludes that dependency.
From the extracted planner directory, supply exact matching files at install time:

```bash
python3.10 scripts/bootstrap_actor_s100.py \
  --model /path/to/actor_9700.onnx \
  --planner-model /path/to/planner_sonic.onnx \
  --bridge-model /path/to/bridge_map.onnx
```

The installer checks size and SHA256 before copying any supplied asset. The
planner also checks both assets before requesting robot control. A normal actor
installation can omit planner and bridge models; the optional planner then
cannot start until they are provisioned with a later installer run. Use a fresh
package directory for different pinned model identities.

## Checks and selection

Run the normal actor checks and confirm Pico input in a read-only session.
The tracked binding source and bundled prebuilt binding expose atomic
right-stick and R3 snapshots; `--build-pico` can rebuild the binding on S100.
The planner is enabled only for SONIC `live` with manual Pico input:

```bash
bash scripts/start_actor_pico.sh sonic live \
  --planner-config config/actor_sonic_planner.json \
  --manual-pico --wait-pico --confirm-motion \
  --robot-id '<this S100 IPv4>' \
  --robot-config /absolute/path/to/running/effective.yaml
```

The live command requires the existing on-site motion preparation and robot
state checks described in [the actor guide](../../README_ACTOR_PICO.md#supported-external-trial).
R3 switches between full-body Pico tracking and joystick legs through a stand
transition. In joystick mode, right-stick Y requests forward/backward speed and
right-stick X requests turning; the stick must return to center before arming.
`--seconds 0` is the live default and runs until STOP or a safety exit.

For source and package validation, run
`scripts/verify_actor_delivery.py --host-inference --model
/path/to/actor_9700.onnx`. The host check covers planner command/fusion and
teleop switching invariants with synthetic inputs. It does not measure S100
latency or robot tracking.
