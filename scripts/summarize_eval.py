"""Aggregate complete evaluations across seeds, with sample standard deviation."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def read_metrics(env, output, data):
    if env == "alfworld":
        rows = [json.loads(x) for x in (output / "episodes.jsonl").read_text().splitlines() if x.strip()]
        keyed = {str(Path(r["game_file"]).resolve()): r for r in rows}
        names = ["test_seen", "test_unseen"]
    else:
        rows = []
        for line in (output / "episodes.tsv").read_text().splitlines():
            task, score, turns, done = line.split("\t")
            rows.append(dict(task_id=int(task), score=float(score), rounds=int(turns)))
        keyed = {r["task_id"]: r for r in rows}
        names = ["test"]
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate episodes in {output}")
    expected_all, result = set(), {}
    for split in names:
        manifest = [json.loads(x) for x in (data / env / f"{split}.jsonl").read_text().splitlines()]
        ids = [str(Path(r["game_file"]).resolve()) if env == "alfworld" else r["task_id"] for r in manifest]
        expected_all.update(ids)
        missing = set(ids) - keyed.keys()
        if missing:
            raise ValueError(f"{output}: missing {len(missing)} {split} episodes")
        selected = [keyed[i] for i in ids]
        metrics = {"success_rate": 100 * mean(float(r["won"]) if env == "alfworld" else float(r["score"] >= 1) for r in selected),
                   "turns": mean(r["rounds"] for r in selected)}
        if env == "webshop":
            metrics["score"] = 100 * mean(r["score"] for r in selected)
        result[split] = metrics
    if set(keyed) != expected_all:
        raise ValueError(f"Unexpected task identities in {output}")
    return result


def summarize(env, output, data, seeds):
    per_seed = {str(seed): read_metrics(env, output / f"seed{seed}", data) for seed in seeds}
    aggregate = {}
    first = next(iter(per_seed.values()))
    for split, metrics in first.items():
        aggregate[split] = {}
        for metric in metrics:
            values = [r[split][metric] for r in per_seed.values()]
            aggregate[split][metric] = dict(mean=mean(values), std=stdev(values) if len(values) > 1 else 0.0)
    return dict(environment=env, evaluation_seeds=seeds, per_seed=per_seed, aggregate=aggregate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["alfworld", "webshop"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    result = summarize(args.env, args.output, args.data, args.seeds)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for split, metrics in result["aggregate"].items():
        print(split + ": " + ", ".join(f"{name} {v['mean']:.2f} ± {v['std']:.2f}" for name, v in metrics.items()))
