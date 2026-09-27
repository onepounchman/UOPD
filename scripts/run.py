"""Render portable configs and run training or three-seed teacher-free evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import yaml
from omegaconf import OmegaConf


def _release_root():
    """Locate configs and manifests in a source checkout or an installed wheel."""
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "configs").is_dir():
        return source_root
    installed_root = Path(sysconfig.get_path("data")) / "share" / "uopd"
    if (installed_root / "configs").is_dir():
        return installed_root
    raise RuntimeError("Could not locate the UOPD configs and data files")


ROOT = _release_root()


def setup_environment(data, output):
    os.environ["UOPD_DATA_DIR"] = str(data.expanduser().resolve())
    os.environ["UOPD_OUTPUT_DIR"] = str(output.expanduser().resolve())
    os.environ.setdefault("ALFWORLD_DATA", str(Path.home() / ".cache/alfworld"))
    paths = [str(ROOT)]
    if os.environ.get("WEBSHOP_ROOT"):
        paths.append(str(Path(os.environ["WEBSHOP_ROOT"]).expanduser().resolve()))
    paths.append(os.environ.get("PYTHONPATH", ""))
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, paths))
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def validate_tasks(config):
    inp = config["buffer"]["explorer_input"]
    for taskset in [inp["taskset"], *inp.get("eval_tasksets", [])]:
        path = Path(taskset["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Prepare the task data first: {path}")
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if not rows:
            raise ValueError(f"Empty task file: {path}")
        if taskset.get("is_eval") and taskset["total_steps"] != len(rows):
            raise ValueError(f"Evaluation count does not match {path}")
        for row in rows:
            if "game_file" in row and not Path(row["game_file"]).is_file():
                raise FileNotFoundError(row["game_file"])


def eval_config(env, model, data, output, seed):
    splits = [("test_seen", 140), ("test_unseen", 134)] if env == "alfworld" else [("test", 128)]
    tasksets = [dict(
        name=split, storage_type="file", path=str(data / env / f"{split}.jsonl"),
        split="test", is_eval=True, total_steps=count, batch_size=32,
        task_selector={"selector_type": "sequential", "seed": seed},
        format={"prompt_key": "game_file" if env == "alfworld" else "task_id"},
        rollout_args={"temperature": 0.4},
        workflow_args={"max_env_steps": 50 if env == "alfworld" else 15},
    ) for split, count in splits]
    env_vars = {k: os.environ[k] for k in ["PYTHONPATH", "ALFWORLD_DATA", "VLLM_ENABLE_V1_MULTIPROCESSING", "TOKENIZERS_PARALLELISM"] if k in os.environ}
    for k in ["JAVA_HOME", "JVM_PATH", "LD_LIBRARY_PATH"]:
        if k in os.environ:
            env_vars[k] = os.environ[k]
    env_vars["ALF_EVAL_AUDIT" if env == "alfworld" else "WS_EVAL_PROGRESS"] = str(
        output / ("episodes.jsonl" if env == "alfworld" else "episodes.tsv"))
    workflow = f"eval_{env}_workflow"
    return dict(
        project="UOPD_EVAL", name=f"{env}_seed{seed}", mode="bench",
        checkpoint_root_dir=str(output / "checkpoints"), continue_from_checkpoint=False,
        model=dict(model_path=model, max_prompt_tokens=2048 if env == "alfworld" else 4096, max_response_tokens=512),
        cluster=dict(node_num=1, gpu_per_node=1),
        buffer=dict(batch_size=32, explorer_input=dict(
            taskset=copy.deepcopy(tasksets[0]), eval_tasksets=tasksets,
            default_workflow_type=workflow, default_eval_workflow_type=workflow)),
        explorer=dict(eval_on_startup=True, bench_on_latest_checkpoint=False,
                      runner_per_model=16, max_timeout=3600,
                      rollout_model=dict(engine_num=1, tensor_parallel_size=1,
                                         enable_prefix_caching=False, enforce_eager=True,
                                         dtype="bfloat16", seed=seed, gpu_memory_utilization=0.7,
                                         enable_chunked_prefill=True), env_vars=env_vars),
        synchronizer=dict(sync_method="nccl"), monitor=dict(monitor_type="tensorboard"),
    )


def launch(config, path, dry_run):
    validate_tasks(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Config: {path}", flush=True)
    if not dry_run:
        subprocess.run([sys.executable, "-m", "trinity.cli.launcher", "run", "--config", str(path)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "eval"])
    parser.add_argument("--env", choices=["alfworld", "webshop"], required=True)
    parser.add_argument("--size", choices=["1.5b", "3b"], default="3b")
    parser.add_argument("--method", choices=["opd", "tcod_f2b", "tcod_b2f", "ftb_opd", "uopd"], default="uopd")
    parser.add_argument("--model", help="Student/teacher HF identifier or trained HF checkpoint for evaluation")
    parser.add_argument("--teacher", help="Override the frozen training teacher")
    parser.add_argument("--data", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--dry-run", action="store_true", help="Validate data and write configs without Ray/GPU execution")
    args = parser.parse_args()
    args.data, args.output = args.data.expanduser().resolve(), args.output.expanduser().resolve()
    setup_environment(args.data, args.output)
    if args.command == "train":
        config = OmegaConf.to_container(OmegaConf.load(ROOT / f"configs/{args.env}/{args.size}/{args.method}.yaml"), resolve=True)
        if args.model:
            config["model"]["model_path"] = args.model
        if args.teacher:
            config["explorer"]["auxiliary_models"][0]["model_path"] = args.teacher
        for key in ["JAVA_HOME", "JVM_PATH", "LD_LIBRARY_PATH", "VLLM_ENABLE_V1_MULTIPROCESSING"]:
            if key in os.environ:
                config["explorer"]["env_vars"][key] = os.environ[key]
        if (args.output / "checkpoints").exists():
            raise FileExistsError("Use a new --output directory for a new training run")
        launch(config, args.output / "train.yaml", args.dry_run)
    else:
        if not args.model:
            parser.error("eval requires --model (a checkpoint, student, or RL teacher)")
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("Evaluation seeds must be distinct")
        model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
        for seed in args.seeds:
            out = args.output / f"seed{seed}"
            if any((out / name).exists() for name in ["episodes.jsonl", "episodes.tsv", "checkpoints"]):
                raise FileExistsError(f"Use a fresh evaluation output directory: {out}")
            cfg = eval_config(args.env, model, args.data, out, seed)
            launch(cfg, out / "eval.yaml", args.dry_run)
        if not args.dry_run:
            subprocess.run([sys.executable, str(ROOT / "scripts/summarize_eval.py"),
                            "--env", args.env, "--data", str(args.data), "--output", str(args.output),
                            "--seeds", *map(str, args.seeds)], check=True)


if __name__ == "__main__":
    main()
