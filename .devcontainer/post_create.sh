#! /usr/bin/env bash
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/experiment-config/config-cluster.yaml}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
uv sync --dev --frozen

if [ -d "$REPO_ROOT/.git" ] && [ -f "$REPO_ROOT/.pre-commit-config.yaml" ] && uv run python -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('pre_commit') else 1)"; then
  uv run pre-commit install --install-hooks
fi
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"
export KMP_DUPLICATE_LIB_OK=TRUE

PYTHON_EXEC=${PYTHON_EXEC:-"uv run python"}
REPORTS_DIR=${REPORTS_DIR:-$REPO_ROOT/reports}
OUTPUTS_DIR=${OUTPUTS_DIR:-$REPORTS_DIR/outputs}
RUN_SCRIPTS_DIR="${RUN_SCRIPTS_DIR:-$REPO_ROOT/experiment-config/run}"
POST_CREATE_TIMEOUT="${POST_CREATE_TIMEOUT:-1h}"
DONE_DIR="${DONE_DIR:-$REPORTS_DIR/.post_create_done}"

has_file() {
  local path="$1"
  [ -e "$path" ] && [ -s "$path" ]
}

run_with_timeout() {
  local timeout_spec="$1"
  local script_path="$2"
  TIMEOUT_SPEC="$timeout_spec" SCRIPT_PATH="$script_path" python3 - <<'PY'
import os
import re
import subprocess
import sys

spec = os.environ["TIMEOUT_SPEC"].strip()
path = os.environ["SCRIPT_PATH"]
match = re.fullmatch(r"(\d+)([smhd]?)", spec)
if not match:
    print(f"Unsupported timeout specification: {spec}", file=sys.stderr)
    sys.exit(2)
value = int(match.group(1))
unit = match.group(2) or "s"
timeout_seconds = value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
try:
    completed = subprocess.run(["bash", path], check=False, timeout=timeout_seconds)
    sys.exit(completed.returncode)
except subprocess.TimeoutExpired:
    print(f"Command timed out after {spec}", file=sys.stderr)
    sys.exit(124)
PY
}

run_script_job() {
  local script_path="$1"
  local script_name
  script_name="$(basename "$script_path")"
  local marker="$DONE_DIR/${script_name}.done"

  if has_file "$marker"; then
    echo "Skipping $script_name (already done: $marker)"
    return 0
  fi

  echo "Running $script_name"
  if run_with_timeout "$POST_CREATE_TIMEOUT" "$script_path"; then
    mkdir -p "$DONE_DIR"
    : > "$marker"
    return 0
  fi

  local status=$?
  if [ "$status" -eq 124 ]; then
    echo "Skipping $script_name (timed out after $POST_CREATE_TIMEOUT)"
    return 0
  fi

  echo "$script_name failed with exit code $status" >&2
  return 0
}

for script in "$RUN_SCRIPTS_DIR"/run_*.sh; do
  [ -f "$script" ] || continue
  run_script_job "$script"
done

uv run python "$REPO_ROOT/scripts/utils/problem_benchmark.py" \
  --all-problems \
  --skip-run \
  --output-dir "$REPORTS_DIR"

echo "Summary tables written under $REPORTS_DIR"
