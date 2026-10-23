"""Run frozen cases through shared inference and simulator adapters."""

import argparse
import json
from pathlib import Path

from holomotion.src.utils.acceptance_profile import load_acceptance_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["mujoco", "isaac"], required=True
    )
    parser.add_argument(
        "--task", choices=["velocity", "motion"], required=True
    )
    parser.add_argument("--stage", choices=["pilot", "final"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--formal-data-root", type=Path)
    args = parser.parse_args()
    extra = {}
    if args.task == "motion":
        if args.backend == "isaac":
            if args.formal_data_root is None:
                parser.error("Motion Isaac requires --formal-data-root")
            extra["formal_data_root"] = args.formal_data_root.resolve()
            from holomotion.src.evaluation.acceptance_motion_isaac import (
                run_motion_acceptance as run_acceptance,
            )
        else:
            from holomotion.src.evaluation.acceptance_motion_mujoco import (
                run_motion_acceptance as run_acceptance,
            )
    elif args.backend == "mujoco":
        from holomotion.src.evaluation.acceptance_mujoco import (
            run_velocity_acceptance as run_acceptance,
        )
    else:
        from holomotion.src.evaluation.acceptance_isaac import (
            run_velocity_acceptance as run_acceptance,
        )

    result = run_acceptance(
        model_path=args.model.resolve(),
        manifest=json.loads(args.manifest.read_text()),
        profile=load_acceptance_profile(args.profile),
        stage=args.stage,
        output=args.output.resolve(),
        record_video=not args.no_video,
        **extra,
    )
    print(json.dumps(result["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
