#!/usr/bin/env bash
#
# Convenience wrapper around pipeline.py.
# Forwards all args to the underlying CLI and surfaces where reports landed.
#
# Examples:
#   ./run.sh --help
#   ./run.sh --tickers AAPL --dry-run
#   ./run.sh --tickers AAPL,MSFT,NVDA --max-tickers 2
#   ./run.sh --ticker-file watchlist.txt
#   ./run.sh --max-tickers 5                           # screen Finviz, top 5
#   ./run.sh --filter-overrides "Sector=Technology"
#   ./run.sh --rerun-today --tickers AAPL              # retry today's failed
#

set -euo pipefail

# Always run from the repo root so config.py / .env / results/ resolve.
cd "$(dirname "$(readlink -f "$0")")"

# Pick a Python interpreter:
#   1. Honour an explicit override:        PYTHON=/path/to/python ./run.sh ...
#   2. Project virtualenv at .venv/        (created with `python -m venv .venv`)
#   3. System python3
# Each candidate is probed for a critical import so we don't silently use a
# half-built venv. (We've been bitten by an empty .venv left behind by a
# failed `uv sync`.) Skip `uv run` for the same reason.
_probe() {
    "$1" -c "import dotenv, langchain_core" >/dev/null 2>&1
}

if [[ -n "${PYTHON:-}" ]]; then
    if ! _probe "$PYTHON"; then
        echo "[run.sh] error: PYTHON=$PYTHON cannot import 'dotenv' / 'langchain_core'." >&2
        echo "[run.sh] Install deps with: $PYTHON -m pip install -e ." >&2
        exit 1
    fi
elif [[ -x .venv/bin/python ]] && _probe .venv/bin/python; then
    PYTHON=.venv/bin/python
elif _probe python3; then
    PYTHON=python3
else
    echo "[run.sh] error: no Python interpreter has the required deps installed." >&2
    echo "[run.sh] Install with:  pip install -e ." >&2
    echo "[run.sh] Or override:   PYTHON=/path/to/python ./run.sh ..." >&2
    exit 1
fi

# Friendly check: pipeline needs *some* LLM API key. .env or shell env both fine.
if [[ ! -f .env ]] \
   && [[ -z "${OPENROUTER_API_KEY:-}${OPENAI_API_KEY:-}${ANTHROPIC_API_KEY:-}${GOOGLE_API_KEY:-}" ]]; then
    echo "[run.sh] warning: no .env file and no LLM API key in environment." >&2
    echo "[run.sh] Set OPENROUTER_API_KEY (or your provider's key) before running." >&2
fi

# Run the pipeline. Args pass through verbatim.
$PYTHON pipeline.py "$@"
status=$?

# After a successful, non-dry-run analyze, show what was written today.
is_dry_run=0
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && is_dry_run=1 && break
done

if [[ $status -eq 0 ]] && [[ $is_dry_run -eq 0 ]]; then
    today=$(date +%Y%m%d)
    if [[ -d "results/by_date/$today" ]]; then
        echo ""
        echo "[run.sh] Reports written today (results/by_date/$today/):"
        ls -lh "results/by_date/$today/" | tail -n +2
    fi
fi

exit $status
