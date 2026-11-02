#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")/.."
BASE="$HOME/AI6130-Ass2-G2605698H"
module load python/3.12.1-gcc11
module load cuda/12.8.1
export HF_HOME="$BASE/hf-cache" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$BASE/pip-cache" TMPDIR="$BASE/tmp" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
mkdir -p "$BASE" artifacts logs
if [ ! -x "$BASE/env/bin/python" ]; then
 python3 -m venv "$BASE/env"
fi
inherited="$HOME/Nuosu-Language-Model/.venv/lib/python3.12/site-packages"
if [ ! -d "$inherited/torch" ]; then echo "Reusable PyTorch directory missing" >&2; exit 1; fi
site="$BASE/env/lib/python3.12/site-packages"
printf '%s\n' "$inherited" > "$site/inherited_torch.pth"
PY="$BASE/env/bin/python"
"$PY" -m pip install --disable-pip-version-check --progress-bar off -r requirements-compatible.txt
"$PY" -m pip install --disable-pip-version-check --no-deps ../AI6130_Assignment2/peft
"$PY" -m pip freeze > artifacts/nscc-requirements-resolved.txt
"$PY" tools/audit.py
"$PY" tools/test_protocol.py
"$PY" tools/download_models.py
"$PY" -c 'import json,sys,importlib.metadata as m; from pathlib import Path; Path("artifacts/nscc-environment.json").write_text(json.dumps({"python":sys.version,"versions":{n:m.version(n) for n in ["torch","transformers","accelerate","datasets"]}},indent=2))'
echo PREPARATION_COMPLETE
