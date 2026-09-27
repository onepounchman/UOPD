"""Materialize portable task manifests without changing their order."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(environment, output, alfworld_root=None, manifests=None):
    source = Path(manifests or ROOT / "data/manifests") / environment
    destination = Path(output) / environment
    pending = {}
    for path in sorted(source.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        identities = set()
        for row in rows:
            key = row["game_file"] if environment == "alfworld" else row["task_id"]
            if key in identities:
                raise ValueError(f"Duplicate task in {path}: {key}")
            identities.add(key)
            if environment == "alfworld":
                if alfworld_root is None:
                    raise ValueError("--alfworld-root must point to the alfworld-download directory")
                relative = Path(row["game_file"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Expected portable game path: {relative}")
                game = Path(alfworld_root).expanduser().resolve() / relative
                if not game.is_file():
                    raise FileNotFoundError(game)
                row["game_file"] = str(game)
        pending[path.name] = rows
    if not pending:
        raise FileNotFoundError(f"No manifests in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in pending.items():
        (destination / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"{environment}/{name}: {len(rows)} tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["alfworld", "webshop", "both"], default="both")
    parser.add_argument("--output", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--alfworld-root", type=Path)
    args = parser.parse_args()
    for env in (["alfworld", "webshop"] if args.env == "both" else [args.env]):
        prepare(env, args.output, args.alfworld_root)
