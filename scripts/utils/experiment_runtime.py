from __future__ import annotations

import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(os.environ.get("REPO_ROOT", "/Users/xiaoyuhe/Causal-LLM"))
PYTHON_EXEC = os.environ.get("PYTHON_EXEC", "python3")
RAW_ROOT = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
REPORTS_ROOT = PROJECT_ROOT / "reports"
OUTPUTS_ROOT = REPORTS_ROOT / "outputs"
SOLVER_CONFIG_DIR = PROJECT_ROOT / "experiment-config" / "solvers"
GUIDE_ROOT = PROJECT_ROOT / "scripts" / "baseline" / "GUIDE"
KCRL_ROOT = PROJECT_ROOT / "scripts" / "baseline" / "KCRL"
UNIFIED_ROOT = PROJECT_ROOT / "scripts" / "baseline" / "Causal-LLM_Unified_One-Shot_Framework" / "Dag_generation and model_evaluation"
CAMML_ROOT = PROJECT_ROOT / "scripts" / "baseline" / "CaMML"
MINOBSX_ROOT = PROJECT_ROOT / "scripts" / "baseline" / "MINOBSx"
ILSCSL_ROOT = PROJECT_ROOT / "scripts" / "baseline" / "ILS-CSL"
JTGAP_ROOT = PROJECT_ROOT / "scripts" / "baseline" / "JunctionTreeGap"
DEFAULT_ILSCSL_SIZE = "1000"
DEFAULT_CAMML_SIZE = "1000"
DEFAULT_MINOBSX_SIZE = "1000"
BN_PARENT_LIMIT = 3
BN_DISCRETE_BINS = 4
BN_EQUIVALENT_SAMPLE_SIZE = 10
UNIFIED_FAMILY_PROBLEMS = {"BIAS", "LEGAL"}

METHOD_OUTPUT_DIR_NAMES = {
    "DataAssist": "DataAssist",
    "GUIDE": "GUIDE",
    "KCRL": "KCRL",
    "UnifiedOneShot": "UnifiedOneShot",
    "LLM_CaMML": "LLM_CaMML",
    "LLM_MINOBSx": "LLM_MINOBSx",
    "ILSCSL": "ILSCSL",
    "JunctionTreeGap": "JunctionTreeGap",
}


def raw_problem_path(problem: str, *parts: str) -> Path:
    return RAW_ROOT / problem / Path(*parts)


def processed_problem_path(problem: str, *parts: str) -> Path:
    return PROCESSED_ROOT / problem / Path(*parts)


def method_output_dir(method: str, problem: str, *parts: str) -> Path:
    base = OUTPUTS_ROOT / METHOD_OUTPUT_DIR_NAMES.get(method, method) / problem
    return base / Path(*parts) if parts else base


def quote(path: Path | str) -> str:
    import shlex

    return shlex.quote(str(path))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return config


def load_solver_config(name: str) -> dict[str, object]:
    path = SOLVER_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    config = load_config(path)
    config.pop("name", None)
    config.pop("method", None)
    return config


def solver_value(config: dict[str, object], key: str, default: object) -> str:
    value = config.get(key, default)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if value is None:
        return str(default)
    return str(value)


def solver_size_for_problem(config: dict[str, object], problem: str, default: str) -> str:
    sizes = config.get("sizes", {})
    if isinstance(sizes, dict):
        value = sizes.get(problem, default)
        if value is None:
            return default
        return str(value)
    return default

