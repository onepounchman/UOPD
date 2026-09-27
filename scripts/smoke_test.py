"""Run a short end-to-end training smoke test with a released configuration."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from scripts.run import ROOT, launch, setup_environment


METHODS = ("opd", "tcod_f2b", "tcod_b2f", "ftb_opd", "uopd")


def smoke_config(env: str, size: str, method: str, data: Path, output: Path, steps: int):
    """Resolve one release config and shorten only its run length."""
    setup_environment(data, output)
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / f"configs/{env}/{size}/{method}.yaml"), resolve=True
    )
    config["project"] = "UOPD_SMOKE"
    config["name"] = f"{env}_{size}_{method}_smoke"
    config["continue_from_checkpoint"] = False
    config["buffer"]["total_steps"] = steps
    config["trainer"]["total_steps"] = steps
    config["trainer"]["save_interval"] = steps
    config["explorer"]["eval_interval"] = steps + 1
    config["explorer"]["eval_on_startup"] = False
    workflow_args = config["buffer"]["explorer_input"]["taskset"].get("workflow_args") or {}
    if "total_steps" in workflow_args:
        workflow_args["total_steps"] = steps
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["alfworld", "webshop"], required=True)
    parser.add_argument("--size", choices=["1.5b", "3b"], default="3b")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["uopd"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--data", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--teacher")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    data = args.data.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    for method in args.methods:
        output = output_root / f"{args.env}-{args.size}-{method}"
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Use a fresh smoke-test output directory: {output}")
        config = smoke_config(args.env, args.size, method, data, output, args.steps)
        if args.model:
            config["model"]["model_path"] = args.model
        if args.teacher:
            config["explorer"]["auxiliary_models"][0]["model_path"] = args.teacher
        launch(config, output / "smoke.yaml", args.dry_run)


if __name__ == "__main__":
    main()
