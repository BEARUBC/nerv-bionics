#!/usr/bin/env bash
# Launch the test bench without having to know which Python has the libraries.
#
# This machine has several Python installations and only some of them have numpy,
# scikit-learn and pyriemann. Running the wrong one fails with ModuleNotFoundError
# and no server starts, which looks like "connection refused" in the browser.
# This script finds an interpreter that actually works and uses that one.
#
#   ./run_ui.sh                 # browser test bench on http://127.0.0.1:8000
#   ./run_ui.sh --port 8001     # any classification.server flag is passed through

set -euo pipefail

cd "$(dirname "$0")"

REQUIRED="numpy scipy sklearn pyriemann"

find_python() {
  local candidates=(
    "${NERV_PYTHON:-}"
    "$(command -v python || true)"
    "$(command -v python3 || true)"
    "$HOME/opt/anaconda3/bin/python3"
    "$HOME/anaconda3/bin/python3"
    "$HOME/miniconda3/bin/python3"
    /opt/homebrew/bin/python3
    /usr/local/bin/python3
  )

  for candidate in "${candidates[@]}"; do
    [ -z "$candidate" ] && continue
    [ -x "$candidate" ] || continue
    if "$candidate" -c "import ${REQUIRED// /, }" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

if ! PYTHON="$(find_python)"; then
  echo "Could not find a Python with the required packages ($REQUIRED)." >&2
  echo >&2
  echo "Install them into the interpreter you want to use:" >&2
  echo "    python -m pip install -r requirements.txt" >&2
  echo >&2
  echo "Or point this script at one:" >&2
  echo "    NERV_PYTHON=/path/to/python ./run_ui.sh" >&2
  exit 1
fi

echo "Using $PYTHON"
exec "$PYTHON" -u -m classification.server "$@"
