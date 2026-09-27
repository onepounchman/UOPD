#!/usr/bin/env bash
# Run inside the Python environment used for UOPD; Java must already be available.
set -euo pipefail
uopd_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
webshop_root=${WEBSHOP_ROOT:-"$uopd_root/third_party/WebShop"}
revision=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
if [[ ! -d "$webshop_root" ]]; then
  git clone https://github.com/princeton-nlp/WebShop.git "$webshop_root"
  git -C "$webshop_root" checkout "$revision"
elif [[ $(git -C "$webshop_root" rev-parse HEAD) != "$revision" ]]; then
  echo "Expected WebShop revision $revision; choose a fresh WEBSHOP_ROOT." >&2
  exit 1
fi
export WEBSHOP_ROOT="$webshop_root"
export PYTHONPATH="$webshop_root:${PYTHONPATH:-}"
# Text environments do not need Selenium/browser dependencies.
python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['WEBSHOP_ROOT']) / 'web_agent_site/envs/__init__.py'
s = p.read_text()
line = 'from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv'
if '\n' + line in '\n' + s and 'try:\n    ' + line not in s:
    s = s.replace(line, 'try:\n    ' + line + '\nexcept ImportError:\n    WebAgentSiteEnv = None')
p.write_text(s)
PY
python -m spacy download en_core_web_lg
mkdir -p "$webshop_root/data" "$webshop_root/search_engine/resources_1k"
cd "$webshop_root/data"
[[ -f items_shuffle_1000.json ]] || gdown --id 1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib -O items_shuffle_1000.json
[[ -f items_ins_v2_1000.json ]] || gdown --id 1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu -O items_ins_v2_1000.json
[[ -f items_human_ins.json ]] || gdown --id 14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O -O items_human_ins.json
cd "$webshop_root/search_engine"
python - <<'PY'
import json
from pathlib import Path
from web_agent_site.engine.engine import load_products
from web_agent_site.utils import BASE_DIR
products, *_ = load_products(filepath=str(Path(BASE_DIR).parent / 'data/items_shuffle_1000.json'))
with open('resources_1k/documents.jsonl', 'w') as output:
    for p in products[:1000]:
        options = ', and '.join(f"{k}: {', '.join(v)}" for k, v in p.get('options', {}).items())
        text = ' '.join([p['Title'], p['Description'], p['BulletPoints'][0], options]).lower()
        output.write(json.dumps({'id': p['asin'], 'contents': text, 'product': p}) + '\n')
PY
python -m pyserini.index.lucene --collection JsonCollection \
  --input resources_1k --index indexes_1k --generator DefaultLuceneDocumentGenerator \
  --threads 1 --storePositions --storeDocvectors --storeRaw
echo "WebShop ready. Export WEBSHOP_ROOT=$webshop_root before running UOPD."
