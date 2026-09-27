# Tasks and teacher actions

| Environment | Train tasks | Nonempty expert action sequences | Test tasks |
|---|---:|---:|---:|
| ALFWorld | 3,553 | 3,296 | 140 seen, 134 unseen |
| WebShop | 6,410 | 4,651 | 128 |

`train_expert.jsonl` keeps all training tasks. Expert actions come from one
greedy rollout of the environment-specific GiGPO 7B teacher per task. Successful
trajectories were replay-verified. Unsuccessful tasks have `actions: []`; B2F
starts the student at the initial state for these tasks. This is distinct from
dropping failed tasks from the training distribution.

ALFWorld `train_legacy.jsonl` and `train.jsonl` contain the same game identities
in different historical orders. OPD/F2B use the legacy manifest; the newer B2F,
FTB-OPD and UOPD configurations use the newer manifest. The configurations make
their selectors explicit. A fixed random seed does not make differently ordered
lists produce the same sampled task sequence.

WebShop training IDs are 500–6909. The fixed 128-task test manifest selects IDs
from the held-out pool 0–499; use this manifest rather than the first 128 IDs.
`test_pool.jsonl` retains the full 500-task pool for periodic training-time checks. OPD/F2B/FTB-OPD/UOPD
configs use sequential training-task selection. The Table 1 B2F run used random
selection; its config retains that setting.

The JSONL files contain task identities and actions, not the underlying game
or product assets. `prepare_data.py` resolves ALFWorld paths against your local
download and keeps the row order unchanged.

## Recollect experts (optional)

The included actions can be used directly. To collect a fresh dataset, each
worker uses one visible GPU and a disjoint shard. For example, two workers:

```bash
export PYTHONPATH="$PWD:${WEBSHOP_ROOT:-}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=0 python scripts/collect_experts.py --env alfworld \
  --tasks data/prepared/alfworld/train.jsonl --output outputs/experts-alfworld \
  --rank 0 --world-size 2
CUDA_VISIBLE_DEVICES=1 python scripts/collect_experts.py --env alfworld \
  --tasks data/prepared/alfworld/train.jsonl --output outputs/experts-alfworld \
  --rank 1 --world-size 2
```

Launch these commands in separate processes or allocations for parallel collection.
Use the same `--world-size` for every rank. Every task gets one attempt, temperature
0, at most 512 generated tokens per action, and 50/15 turns for ALFWorld/WebShop.
Task-derived generation seeds are independent of the worker rank. A shard is marked
complete only after all its tasks finish and successful actions pass replay checks.

```bash
python scripts/merge_experts.py --tasks data/prepared/alfworld/train.jsonl \
  --collection outputs/experts-alfworld --output outputs/train_expert.jsonl
```

The merger requires exactly one attempt per task, rejects incomplete/duplicate
shards, and preserves failed tasks with empty action lists. Use the merged file
as `alfworld/train_expert.jsonl` under a separate prepared-data root, selected by
`scripts/run.py --data`. Use `--env webshop` and the corresponding paths for WebShop.

Collected teacher revisions:

- ALFWorld: `895c9fe1db239ae6140ffec7801a51e157292f83`
- WebShop: `8e58a29cdbad307671427340b1fe06a15ce05eeb`

The optional recollection command accepts `--teacher` to use a local snapshot.
