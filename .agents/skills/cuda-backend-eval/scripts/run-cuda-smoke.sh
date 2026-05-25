#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${repo_root}"

if [[ ! -x .venv/bin/python ]]; then
  echo "missing .venv/bin/python; create the project-local venv first" >&2
  exit 1
fi

.venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py -q "$@"
