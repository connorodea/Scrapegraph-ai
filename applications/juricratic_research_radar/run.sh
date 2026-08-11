#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python -m pip install -e .
python -m pip install -r applications/juricratic_research_radar/requirements.txt
exec streamlit run applications/juricratic_research_radar/app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}"
