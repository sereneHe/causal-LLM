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

if [ -d "$REPO_ROOT/.git" ] && uv run python -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('pre_commit') else 1)"; then
  uv run pre-commit install --install-hooks
fi

eval "$(
  uv run python - "$CONFIG_PATH" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).expanduser().resolve()
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

runtime = config.get("runtime", {}) or {}
methods = config.get("methods", {}) or {}

def emit(name: str, value) -> None:
    if isinstance(value, bool):
        value = "TRUE" if value else "FALSE"
    print(f'export {name}={shlex.quote(str(value))}')

python_exec = runtime.get("python_exec", "python3")
emit("PYTHON_EXEC", f"uv run {python_exec}")
emit("REPORTS_DIR", runtime.get("reports_dir", str(config_path.parent.parent / "reports")))
emit("OUTPUTS_DIR", runtime.get("outputs_dir", str(config_path.parent.parent / "reports" / "outputs")))
emit("TRACKING", config.get("tracking", "mlflow"))
emit("PLOT", config.get("plot", False))
emit("PLOT_DPI", config.get("plot_dpi", 300))
emit("SKIP_FIRST", config.get("skip_first", 0))

if runtime.get("kmp_duplicate_lib_ok", False):
    emit("KMP_DUPLICATE_LIB_OK", True)

for method_name, method_cfg in methods.items():
    if not isinstance(method_cfg, dict):
        continue
    prefix = method_name.upper()
    if "enabled" in method_cfg:
        emit(f"{prefix}_ENABLED", method_cfg["enabled"])
    if "problem" in method_cfg:
        emit(f"{prefix}_PROBLEM", method_cfg["problem"])
    if "output_dir" in method_cfg:
        emit(f"{prefix}_OUTPUT_DIR", method_cfg["output_dir"])
    for key, value in method_cfg.items():
        if key in {"enabled", "problem", "output_dir"}:
            continue
        emit(f"{prefix}_{key.upper()}", value)
PY
)"

run_summary() {
  problem="$1"
  methods="$2"
  uv run python "$REPO_ROOT/scripts/utils/problem_benchmark.py" \
    --problem "$problem" \
    --methods "$methods" \
    --skip-run \
    --output-dir "$REPORTS_DIR"
}

if [ "${DATAASSIST_ENABLED:-FALSE}" = "TRUE" ]; then
  OUTPUT_DIR="${DATAASSIST_OUTPUT_DIR}" sh "$REPO_ROOT/experiment-config/run/run_DataAssist.sh"
  run_summary "${DATAASSIST_PROBLEM}" "DataAssist"
fi

if [ "${GUIDE_ENABLED:-FALSE}" = "TRUE" ]; then
  OUTPUT_DIR="${GUIDE_OUTPUT_DIR}" \
  NUM_EPOCHS="${GUIDE_NUM_EPOCHS}" \
  HIDDEN_DIM="${GUIDE_HIDDEN_DIM}" \
  NHEADS="${GUIDE_NHEADS}" \
  ACTOR_LR="${GUIDE_ACTOR_LR}" \
  PRIOR_FRACTION="${GUIDE_PRIOR_FRACTION}" \
  sh "$REPO_ROOT/experiment-config/run/run_GUIDE.sh"
  run_summary "${GUIDE_PROBLEM}" "GUIDE"
fi

if [ "${KCRL_ENABLED:-FALSE}" = "TRUE" ]; then
  OUTPUT_DIR="${KCRL_OUTPUT_DIR}" \
  NB_EPOCH="${KCRL_NB_EPOCH}" \
  INPUT_DIMENSION="${KCRL_INPUT_DIMENSION}" \
  LAMBDA_ITER_NUM="${KCRL_LAMBDA_ITER_NUM}" \
  sh "$REPO_ROOT/experiment-config/run/run_KCRL.sh"
  run_summary "${KCRL_PROBLEM}" "KCRL"
fi

if [ "${UNIFIEDONESHOT_ENABLED:-FALSE}" = "TRUE" ]; then
  OUTPUT_DIR="${UNIFIEDONESHOT_OUTPUT_DIR}" \
  ENABLED_MODELS="${UNIFIEDONESHOT_ENABLED_MODELS}" \
  sh "$REPO_ROOT/experiment-config/run/run_UnifiedOneShot.sh"
  run_summary "${UNIFIEDONESHOT_PROBLEM}" "UnifiedOneShot"
fi

echo "Summary tables written under $REPORTS_DIR"
