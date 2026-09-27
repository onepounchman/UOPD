"""Merge one attempt per task; retain failed tasks with empty expert actions."""
import argparse
import json
from pathlib import Path


def merge(tasks_path, collection, output):
    tasks = [json.loads(x) for x in tasks_path.read_text().splitlines() if x.strip()]
    key = "game_file" if "game_file" in tasks[0] else "task_id"
    expected = {r[key] for r in tasks}
    if len(expected) != len(tasks):
        raise ValueError("Task manifest contains duplicates")
    attempts, experts = {}, {}
    shards = sorted((collection / "shards").iterdir())
    if not shards:
        raise ValueError("No collection shards")
    for shard in shards:
        summary = json.loads((shard / "summary.json").read_text())
        if not summary.get("complete"):
            raise ValueError(f"Incomplete shard: {shard}")
        for filename, dest in [("attempts.jsonl", attempts), ("teacher_actions.jsonl", experts)]:
            for line in (shard / filename).read_text().splitlines():
                row = json.loads(line)
                if row[key] in dest:
                    raise ValueError(f"Duplicate {filename} task: {row[key]}")
                dest[row[key]] = row
    if set(attempts) != expected:
        raise ValueError("Attempt identities must match the entire task manifest")
    successful = {k for k, r in attempts.items() if r["success"]}
    if set(experts) != successful:
        raise ValueError("Successful attempts and replay-verified experts differ")
    rows = []
    for task in tasks:
        expert = experts.get(task[key])
        if expert and (not expert["replay_verified"] or expert["score"] < 1 or not expert["actions"]):
            raise ValueError(f"Invalid expert: {task[key]}")
        rows.append({**task, "actions": expert["actions"] if expert else []})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return dict(tasks=len(tasks), successful=len(experts), empty_actions=len(tasks) - len(experts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.tasks, args.collection, args.output), indent=2))
