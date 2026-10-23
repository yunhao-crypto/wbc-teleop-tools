# GMR-free deployment model assets

The GMR-free deployment uses two external model packages. Do not commit model
weights into this source directory.

Each package must have this layout:

```text
MODEL_DIR/
├── config.yaml
└── exported/
    └── exactly_one_model.onnx
```

Pass both directories explicitly:

```bash
bash start_orin_local_pico.sh MOTION_MODEL_DIR VELOCITY_MODEL_DIR
```

Run `scripts/preflight_gmr_free.sh` with the same directories before any
command that starts the robot control stack.
