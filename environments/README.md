# Environment setup

Install the Python dependencies in the root README first. Both benchmarks use
text interactions with the same prompts during training and evaluation.

## ALFWorld

```bash
export ALFWORLD_DATA="$HOME/.cache/alfworld"
alfworld-download
python scripts/prepare_data.py --env alfworld --alfworld-root "$ALFWORLD_DATA"
```

The prepared files point to `json_2.1.1/{train,valid_seen,valid_unseen}/.../game.tw-pddl`.
This setup uses the TextWorld backend; a Unity rendering server is unnecessary.

## WebShop

Install Java (OpenJDK 11 is supported by the pinned Pyserini version), then expose
its JVM to Python. For example, in the active conda environment:

```bash
conda install -c conda-forge openjdk=11 -y
export JAVA_HOME="$CONDA_PREFIX"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"
export WEBSHOP_ROOT="$PWD/third_party/WebShop"
bash environments/setup_webshop.sh
python scripts/prepare_data.py --env webshop
```

The setup script pins WebShop commit
`64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`, downloads the **1,000-product** catalog
and attributes, and builds `search_engine/indexes_1k`. It makes the browser-only
Selenium import optional. It does not install the upstream legacy requirements,
which would replace the current PyTorch stack.

Before running training/evaluation:

```bash
export PYTHONPATH="$PWD:$WEBSHOP_ROOT:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

The workflow selects `observation_mode="text"`, `num_products=1000`, and
`human_goals=False` explicitly. The 1,000 products are separate from the number
of task queries (6,410 train, 128 test).
