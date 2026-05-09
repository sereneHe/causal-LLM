#!/usr/bin/env python3
"""Dispatch causal-LLM baselines from Hydra-style CLI overrides."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from functools import partial
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("REPO_ROOT", "/Users/xiaoyuhe/Causal-LLM"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.experiment_runtime import (
    BN_EQUIVALENT_SAMPLE_SIZE,
    BN_PARENT_LIMIT,
    CAMML_ROOT,
    DEFAULT_CAMML_SIZE,
    DEFAULT_MINOBSX_SIZE,
    DEFAULT_ILSCSL_SIZE,
    GUIDE_ROOT,
    ILSCSL_ROOT,
    JTGAP_ROOT,
    KCRL_ROOT,
    MINOBSX_ROOT,
    PROJECT_ROOT,
    PROCESSED_ROOT,
    PYTHON_EXEC,
    RAW_ROOT,
    REPORTS_ROOT,
    UNIFIED_FAMILY_PROBLEMS,
    UNIFIED_ROOT,
    load_config,
    load_solver_config,
    method_output_dir,
    processed_problem_path,
    quote,
    solver_size_for_problem,
    solver_value,
)


ILSCSL_DATASET_SIZES = {"asia": ["250", "1000"], "child": ["500", "2000"], "insurance": ["500", "2000"], "alarm": ["1000", "4000"], "cancer": ["250", "1000"], "mildew": ["8000", "32000"], "water": ["1000", "4000"], "barley": ["2000", "8000"]}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute causal-LLM baselines from Hydra-style overrides."
    )
    parser.add_argument("--multirun", action="store_true")
    parser.add_argument("--config-name", default=os.environ.get("CONFIG_NAME", "config"))
    parser.add_argument("--step", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if not args.multirun:
        parser.error("Only --multirun mode is supported.")
    return args


def parse_overrides(items: list[str]) -> dict[str, list[str]]:
    overrides: dict[str, list[str]] = {}
    list_value_keys = {"unified.enabled_models"}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")
        if key in list_value_keys:
            overrides[key] = [value]
            continue
        values = [
            part.strip().strip("'").strip('"')
            for part in value.split(",")
            if part.strip()
        ]
        overrides[key] = values or [""]
    return overrides


def expand_override_grid(overrides: dict[str, list[str]]) -> list[dict[str, str]]:
    if not overrides:
        return [{}]
    keys = list(overrides)
    values_product = product(*(overrides[key] for key in keys))
    return [dict(zip(keys, combo)) for combo in values_product]


def select_commands(commands: list[tuple[str, str]], step: str) -> list[tuple[int, str, str]]:
    if step == "all":
        return [(index, label, command) for index, (label, command) in enumerate(commands, start=1)]
    index = int(step)
    if index < 1 or index > len(commands):
        raise ValueError(f"--step must be between 1 and {len(commands)}")
    label, command = commands[index - 1]
    return [(index, label, command)]


def run_with_timeout(command: str, timeout_spec: str) -> int:
    """Run a shell command with a compact timeout specification.

    Supports integers with optional suffixes:
    - s: seconds
    - m: minutes
    - h: hours
    - d: days
    """

    spec = timeout_spec.strip()
    match = re.fullmatch(r"(\d+)([smhd]?)", spec)
    if not match:
        print(f"Unsupported timeout specification: {spec}", file=sys.stderr)
        return 2

    value = int(match.group(1))
    unit = match.group(2) or "s"
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    timeout_seconds = value * multiplier

    try:
        completed = subprocess.run(command, shell=True, check=False, timeout=timeout_seconds)
        return completed.returncode
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {spec}", file=sys.stderr)
        return 124


POST_CREATE_DEFAULT_METHODS = ["DataAssist", "GUIDE", "KCRL", "UnifiedOneShot", "LLM_CaMML", "LLM_MINOBSx", "ILSCSL", "JunctionTreeGap"]


def post_create_list_problems(repo_root: Path, problems_csv: str | None = None) -> list[str]:
    if problems_csv: return [item.strip() for item in problems_csv.split(",") if item.strip()]
    problems_dir = repo_root / "experiment-config" / "problems"
    return sorted(path.stem for path in problems_dir.glob("*.yaml"))


def post_create_list_methods(methods_csv: str | None = None) -> list[str]:
    if methods_csv: return [item.strip() for item in methods_csv.split(",") if item.strip()]
    return list(POST_CREATE_DEFAULT_METHODS)


def post_create_marker_for(state_dir: Path, method: str, problem: str) -> Path:
    return state_dir / method / f"{problem}.ok"


def post_create_output_marker(reports_dir: Path, method: str, problem: str) -> Path | None:
    outputs_root = reports_dir / "outputs"
    mapping = {
        "DataAssist": outputs_root / "DataAssist" / problem / "pc_causal_matrix_eval.json",
        "GUIDE": outputs_root / "GUIDE" / problem / "results_custom.npy",
        "KCRL": outputs_root / "KCRL" / problem / "final_metrics.json",
        "UnifiedOneShot": outputs_root / "UnifiedOneShot" / problem / "adj_combined_results.json",
        "LLM_CaMML": outputs_root / "LLM_CaMML" / problem / "results.txt",
        "LLM_MINOBSx": outputs_root / "LLM_MINOBSx" / problem / "results.txt",
        "ILSCSL": outputs_root / "ILSCSL" / problem / "HC" / "metrics.csv",
        "JunctionTreeGap": outputs_root / "JunctionTreeGap" / problem / "final_metrics.json",
    }
    return mapping.get(method)


def post_create_existing_success(
    reports_dir: Path,
    state_dir: Path,
    method: str,
    problem: str,
) -> bool:
    marker = post_create_marker_for(state_dir, method, problem)
    if marker.exists() and marker.stat().st_size > 0:
        return True

    output_marker = post_create_output_marker(reports_dir, method, problem)
    if output_marker is not None and output_marker.exists() and output_marker.stat().st_size > 0:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return True

    if method == "UnifiedOneShot" and problem in {"BIAS", "LEGAL"}:
        unified_dir = reports_dir / "outputs" / "UnifiedOneShot" / problem
        if any(unified_dir.rglob("adj_combined_results.json")):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            return True

    if method == "ILSCSL":
        ilscsl_dir = reports_dir / "outputs" / "ILSCSL" / problem
        if any(ilscsl_dir.rglob("metrics.csv")):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            return True

    return False


def post_create_command(
    repo_root: Path,
    python_exec: str,
    run_experiment_path: Path,
    method: str,
    problem: str,
) -> str:
    if method == "DataAssist":
        return (
            f"{python_exec} {shlex.quote(str(run_experiment_path))} --multirun --config-name=config-cluster "
            f"problem='{problem}' problem.generator='DataAssist'"
        )
    if method == "GUIDE":
        return (
            f"{python_exec} {shlex.quote(str(run_experiment_path))} --multirun --config-name=config-cluster "
            f"problem='{problem}' problem.generator='GUIDE' guide.num_epochs='1' guide.hidden_dim='64' "
            f"guide.nheads='8' guide.actor_lr='0.001' guide.prior_fraction='0.25'"
        )
    if method == "KCRL":
        return (
            f"{python_exec} {shlex.quote(str(run_experiment_path))} --multirun --config-name=config-cluster "
            f"problem='{problem}' problem.generator='KCRL' kcrl.nb_epoch='1' kcrl.input_dimension='64' "
            f"kcrl.lambda_iter_num='10'"
        )
    if method == "UnifiedOneShot":
        return (
            f"{python_exec} {shlex.quote(str(run_experiment_path))} --multirun --config-name=config-cluster "
            f"problem='{problem}' problem.generator='UnifiedOneShot' unified.enabled_models='PC,causal_llm' "
            f"unified.causal_llm_backbone='Llama'"
        )
    if method == "LLM_CaMML":
        return (
            f"{python_exec} {shlex.quote(str(run_experiment_path))} --multirun --config-name=config-cluster "
            f"problem='{problem}' problem.generator='LLM_CaMML'"
        )
    if method == "LLM_MINOBSx":
        return (
            f"{python_exec} {shlex.quote(str(run_experiment_path))} --multirun --config-name=config-cluster "
            f"problem='{problem}' problem.generator='LLM_MINOBSx'"
        )
    if method == "ILSCSL":
        return (
            f"{python_exec} {shlex.quote(str(run_experiment_path))} --multirun --config-name=config-cluster "
            f"problem='{problem}' problem.generator='ILSCSL' ilscsl.alg='HC' ilscsl.score='bdeu' "
            f"ilscsl.model='gpt-4' ilscsl.data_index='1' ilscsl.datasize_index='0'"
        )
    if method == "JunctionTreeGap":
        return (
            f"{python_exec} {shlex.quote(str(run_experiment_path))} --multirun --config-name=config-cluster "
            f"problem='{problem}' problem.generator='JunctionTreeGap' jtgap.top_k='3' jtgap.gap_weight='0.5' "
            f"jtgap.sparsity_threshold='0.0' jtgap.edge_threshold='0.0' jtgap.covariance_jitter='1e-6' "
            f"jtgap.gap_metric='sym_kl' jtgap.separator_model='gaussian'"
        )
    raise ValueError(f"Unsupported method: {method}")


def post_create_run(
    *,
    repo_root: Path,
    reports_dir: Path,
    python_exec: str,
    run_experiment_path: Path,
    timeout_spec: str,
    force: bool = False,
    problems: list[str] | None = None,
    methods: list[str] | None = None,
) -> int:
    state_dir = reports_dir / "post_create_state"
    log_dir = reports_dir / "post_create_logs"
    run_log = log_dir / "run_log.tsv"
    failure_log = log_dir / "failures.log"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not run_log.exists():
        run_log.write_text(
            "timestamp\tstatus\tmethod\tproblem\texit_code\tduration_s\tmarker\tcommand\n",
            encoding="utf-8",
        )

    selected_problems = problems or post_create_list_problems(repo_root)
    selected_methods = methods or post_create_list_methods()

    def append_run(row: list[str]) -> None:
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write("\t".join(row) + "\n")

    def append_failure(status: str, method: str, problem: str, exit_code: int, duration_s: int, marker: Path, command: str) -> None:
        with failure_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{_utc_now()}] {method}/{problem} {status} exit={exit_code} duration={duration_s}s\n"
            )
            handle.write(f"marker: {marker}\n")
            handle.write(f"command: {command}\n\n")

    for method in selected_methods:
        for problem in selected_problems:
            marker = post_create_marker_for(state_dir, method, problem)
            command = post_create_command(repo_root, python_exec, run_experiment_path, method, problem)

            if not force and post_create_existing_success(reports_dir, state_dir, method, problem):
                append_run([_utc_now(), "SKIP", method, problem, "0", "0", str(marker), command])
                continue

            start = time.time()
            exit_code = run_with_timeout(command, timeout_spec)
            duration_s = int(time.time() - start)
            if exit_code == 0:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                append_run([_utc_now(), "OK", method, problem, "0", str(duration_s), str(marker), command])
                continue

            status = "TIMEOUT" if exit_code == 124 else "FAILED"
            append_run([_utc_now(), status, method, problem, str(exit_code), str(duration_s), str(marker), command])
            append_failure(status, method, problem, exit_code, duration_s, marker, command)

    return 0


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quote(path: Path | str) -> str:
    return shlex.quote(str(path))


def _load_processed_bundle(problem: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    bundle_path = processed_problem_path(problem, "data.npy")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing processed bundle for {problem}: {bundle_path}")

    bundle = np.load(bundle_path, allow_pickle=True).item()
    x = np.asarray(bundle.get("x"))
    y = bundle.get("y")
    if y is None:
        y = np.zeros((x.shape[1], x.shape[1]), dtype=int)
    else:
        y = np.asarray(y)

    nodes_path = processed_problem_path(problem, "nodes.npy")
    if nodes_path.exists():
        nodes = [str(node) for node in np.load(nodes_path, allow_pickle=True).tolist()]
    else:
        nodes = [f"v{i}" for i in range(x.shape[1])]

    if x.shape[1] != len(nodes):
        raise ValueError(
            f"Processed bundle for {problem} has {x.shape[1]} columns but {len(nodes)} nodes"
        )
    if y.shape != (len(nodes), len(nodes)):
        raise ValueError(
            f"Processed bundle for {problem} has adjacency shape {y.shape}, expected {(len(nodes), len(nodes))}"
        )
    return x, y, nodes


def _graph_headers(problem: str, width: int) -> tuple[list[str], list[str]]:
    full_names = [f"{problem}_{index}" for index in range(width)]
    short_names = [f"v{index}" for index in range(width)]
    return full_names, short_names


def _discretize_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.isna().all():
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)

    unique_count = int(numeric.nunique(dropna=True))
    if unique_count <= 1:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)

    non_null = numeric.dropna()
    if pd.api.types.is_integer_dtype(non_null.dtype) or unique_count <= BN_DISCRETE_BINS:
        codes = pd.factorize(non_null, sort=True)[0]
        encoded = pd.Series(index=series.index, dtype=float)
        encoded.loc[non_null.index] = codes
        return encoded.fillna(0).astype(int)

    bins = min(BN_DISCRETE_BINS, unique_count)
    try:
        encoded = pd.qcut(numeric.fillna(numeric.median()), q=bins, labels=False, duplicates="drop")
        return encoded.fillna(0).astype(int)
    except Exception:
        codes = pd.factorize(numeric.fillna(numeric.median()), sort=True)[0]
        return pd.Series(codes, index=series.index).fillna(0).astype(int)


def _discretize_bundle_frame(x: np.ndarray, columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(np.asarray(x), columns=columns)
    encoded = {column: _discretize_column(frame[column]) for column in columns}
    return pd.DataFrame(encoded, columns=columns)


def _write_bn_assets(
    *,
    problem: str,
    x: np.ndarray,
    y: np.ndarray,
    nodes: list[str],
    camml_csv_name: str,
    minobsx_bdeu_name: str,
) -> pd.DataFrame:
    discrete_df = _discretize_bundle_frame(x, nodes)
    aliases = [f"v{i}" for i in range(len(nodes))]
    staged_df = discrete_df.copy()
    staged_df.columns = aliases

    camml_roots = [CAMML_ROOT, CAMML_ROOT / "BI-CaMML"]
    for root in camml_roots:
        data_dir = root / "data"
        graph_dir = root / "BN_structure"
        mapping_dir = graph_dir / "mappings"
        data_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        mapping_dir.mkdir(parents=True, exist_ok=True)
        staged_df.to_csv(data_dir / camml_csv_name, index=False)
        np.savetxt(graph_dir / f"{problem}_graph.txt", y, fmt="%d")
        (mapping_dir / f"{problem}.mapping").write_text("\n".join(nodes) + "\n", encoding="utf-8")
        (mapping_dir / f"{problem}_abb.mapping").write_text("\n".join(aliases) + "\n", encoding="utf-8")

    minobsx_roots = [MINOBSX_ROOT, MINOBSX_ROOT / "minobsxx"]
    for root in minobsx_roots:
        data_dir = root / "data"
        csv_dir = data_dir / "csv"
        bdeu_dir = data_dir / "bdeu"
        graph_dir = root / "BN_structure"
        mapping_dir = graph_dir / "mappings"
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        bdeu_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        mapping_dir.mkdir(parents=True, exist_ok=True)
        staged_df.to_csv(csv_dir / camml_csv_name, index=False)
        np.savetxt(graph_dir / f"{problem}_graph.txt", y, fmt="%d")
        (mapping_dir / f"{problem}.mapping").write_text("\n".join(nodes) + "\n", encoding="utf-8")
        (mapping_dir / f"{problem}_abb.mapping").write_text("\n".join(aliases) + "\n", encoding="utf-8")

    try:
        from pgmpy.estimators import BDeuScore as BDeu
    except ImportError:  # pragma: no cover - older pgmpy fallback
        from pgmpy.estimators.StructureScore import BDeu  # type: ignore

    bdeu_path = (MINOBSX_ROOT / "minobsxx" / "data" / "bdeu" / minobsx_bdeu_name)
    if not bdeu_path.exists():
        scorer = BDeu(staged_df, equivalent_sample_size=BN_EQUIVALENT_SAMPLE_SIZE)
        bdeu_path.parent.mkdir(parents=True, exist_ok=True)
        with bdeu_path.open("w", encoding="utf-8") as handle:
            handle.write(f"{len(nodes)}\n")
            for var_index, node in enumerate(aliases):
                parent_sets: list[tuple[tuple[int, ...], float]] = []
                others = [idx for idx in range(len(nodes)) if idx != var_index]
                for parent_size in range(0, min(BN_PARENT_LIMIT, len(others)) + 1):
                    for parents in combinations(others, parent_size):
                        parent_names = [aliases[parent] for parent in parents]
                        score = float(scorer.local_score(node, parent_names))
                        parent_sets.append((parents, score))
                handle.write(f"{var_index} {len(parent_sets)}\n")
                for parents, score in parent_sets:
                    parent_text = " ".join(str(parent) for parent in parents)
                    if parent_text:
                        handle.write(f"{score:.10f} {len(parents)} {parent_text}\n")
                    else:
                        handle.write(f"{score:.10f} 0\n")

    return staged_df


def preprocess_command(problem: str) -> str:
    return (
        f"{quote(PYTHON_EXEC)} {quote(PROJECT_ROOT / 'scripts' / 'data' / 'preprocessing.py')} "
        f"--dataset {shlex.quote(problem)} --raw-root {quote(RAW_ROOT)} "
        f"--processed-root {quote(PROCESSED_ROOT)}"
    )


def stage_label(method: str, problem: str, stage: str) -> str:
    return f"[{method}-{problem}-{stage}]"


def build_dataassist_job(problem: str) -> tuple[str, list[tuple[str, str]]]:
    solver = load_solver_config("DataAssist")
    output_dir = method_output_dir("DataAssist", problem)
    commands = [
        (stage_label("DataAssist", problem, "数据处理"), preprocess_command(problem)),
        (
            stage_label("DataAssist", problem, "方法运行"),
            (
                f"{quote(PYTHON_EXEC)} {quote(PROJECT_ROOT / 'scripts' / 'utils' / 'metrics.py')} "
                f"--data-npy {quote(processed_problem_path(problem, 'data.npy'))} "
                f"--output-dir {quote(output_dir)}"
            ),
        ),
    ]
    return f"DataAssist_{problem}", commands


def build_guide_job(problem: str, overrides: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    solver = load_solver_config("GUIDE")
    output_dir = method_output_dir("GUIDE", problem)
    num_epochs = overrides.get("guide.num_epochs", solver_value(solver, "num_epochs", "1"))
    hidden_dim = overrides.get("guide.hidden_dim", solver_value(solver, "hidden_dim", "64"))
    nheads = overrides.get("guide.nheads", solver_value(solver, "nheads", "8"))
    actor_lr = overrides.get("guide.actor_lr", solver_value(solver, "actor_lr", "0.001"))
    prior_fraction = overrides.get("guide.prior_fraction", solver_value(solver, "prior_fraction", "0.25"))
    commands = [
        (stage_label("GUIDE", problem, "数据处理"), preprocess_command(problem)),
        (
            stage_label("GUIDE", problem, "方法运行"),
            (
                f"{quote(PYTHON_EXEC)} {quote(GUIDE_ROOT / 'main.py')} "
                f"--data_path {quote(processed_problem_path(problem, 'X.npy'))} "
                f"--adj_path {quote(processed_problem_path(problem, 'adj.npy'))} "
                f"--output_dir {quote(output_dir)} "
                f"--num_epochs {shlex.quote(num_epochs)} "
                f"--hidden_dim {shlex.quote(hidden_dim)} "
                f"--nheads {shlex.quote(nheads)} "
                f"--actor_lr {shlex.quote(actor_lr)} "
                f"--prior_fraction {shlex.quote(prior_fraction)}"
            ),
        ),
    ]
    return f"GUIDE_{problem}", commands


def build_kcrl_job(problem: str, overrides: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    solver = load_solver_config("KCRL")
    output_dir = method_output_dir("KCRL", problem)
    nb_epoch = overrides.get("kcrl.nb_epoch", solver_value(solver, "nb_epoch", "1"))
    input_dimension = overrides.get("kcrl.input_dimension", solver_value(solver, "input_dimension", "64"))
    lambda_iter_num = overrides.get("kcrl.lambda_iter_num", solver_value(solver, "lambda_iter_num", "10"))
    commands = [
        (stage_label("KCRL", problem, "数据处理"), preprocess_command(problem)),
        (
            stage_label("KCRL", problem, "方法运行"),
            (
                f"KCRL_DATA_PATH={quote(processed_problem_path(problem))} "
                f"KCRL_OUTPUT_DIR={quote(output_dir)} "
                f"KCRL_NB_EPOCH={shlex.quote(nb_epoch)} "
                f"KCRL_INPUT_DIMENSION={shlex.quote(input_dimension)} "
                f"KCRL_LAMBDA_ITER_NUM={shlex.quote(lambda_iter_num)} "
                f"KCRL_READ_DATA=1 "
                f"{quote(PYTHON_EXEC)} {quote(KCRL_ROOT / 'kcrl_demo.py')}"
            ),
        ),
    ]
    return f"KCRL_{problem}", commands


def build_unified_job(problem: str, overrides: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    solver = load_solver_config("UnifiedOneShot")
    output_dir = method_output_dir("UnifiedOneShot", problem)
    enabled_models = overrides.get("unified.enabled_models", solver_value(solver, "enabled_models", "PC"))
    causal_llm_backbone = overrides.get("unified.causal_llm_backbone", solver_value(solver, "causal_llm_backbone", "Llama"))
    if problem in UNIFIED_FAMILY_PROBLEMS:
        family_runner = (
            "from pathlib import Path\n"
            "import subprocess\n"
            "import sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            f"processed = Path({str(processed_problem_path(problem))!r})\n"
            f"output_root = Path({str(output_dir)!r})\n"
            f"method_dir = Path({str(UNIFIED_ROOT)!r})\n"
            f"enabled_models = {enabled_models!r}\n"
            f"causal_llm_backbone = {causal_llm_backbone!r}\n"
            "for variant_dir in sorted(p for p in processed.iterdir() if p.is_dir() and (p / 'data.npy').exists()):\n"
            "    tmp = output_root / variant_dir.name / 'tmp_inputs'\n"
            "    tmp.mkdir(parents=True, exist_ok=True)\n"
            "    payload = np.load(variant_dir / 'data.npy', allow_pickle=True).item()\n"
            "    x = np.asarray(payload['x'])\n"
            "    y = np.asarray(payload['y'])\n"
            "    pd.DataFrame(x).to_csv(tmp / 'samples.csv', index=False)\n"
            "    pd.DataFrame(y).to_csv(tmp / 'adj.csv', index=False, header=False)\n"
            "    cmd = [\n"
            "        sys.executable,\n"
            "        'main_runner.py',\n"
            "        '--ground-truth-path', str(tmp / 'adj.csv'),\n"
            "        '--gaussian-path', str(tmp / 'samples.csv'),\n"
            "        '--output-dir', str(output_root / variant_dir.name),\n"
            "        '--enabled-models', enabled_models,\n"
            "        '--causal-llm-backbone', causal_llm_backbone,\n"
            "    ]\n"
            "    subprocess.run(cmd, cwd=str(method_dir), check=True)\n"
        )
        commands = [
            (stage_label("UnifiedOneShot", problem, "数据处理"), preprocess_command(problem)),
            (stage_label("UnifiedOneShot", problem, "方法运行"), f"{quote(PYTHON_EXEC)} -c {shlex.quote(family_runner)}"),
        ]
    else:
        tmp_dir = output_dir / "tmp_inputs"
        prep_csv = (
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            f"processed = Path({str(processed_problem_path(problem))!r})\n"
            f"tmp = Path({str(tmp_dir)!r})\n"
            "tmp.mkdir(parents=True, exist_ok=True)\n"
            "x = np.load(processed / 'X.npy', allow_pickle=True)\n"
            "y = np.load(processed / 'adj.npy', allow_pickle=True)\n"
            "pd.DataFrame(x).to_csv(tmp / 'samples.csv', index=False)\n"
            "pd.DataFrame(y).to_csv(tmp / 'adj.csv', index=False, header=False)\n"
        )
        commands = [
            (stage_label("UnifiedOneShot", problem, "数据处理"), preprocess_command(problem)),
            (stage_label("UnifiedOneShot", problem, "数据处理"), f"{quote(PYTHON_EXEC)} -c {shlex.quote(prep_csv)}"),
            (
                stage_label("UnifiedOneShot", problem, "方法运行"),
                (
                    f"cd {quote(UNIFIED_ROOT)} && "
                    f"{quote(PYTHON_EXEC)} main_runner.py "
                    f"--ground-truth-path {quote(tmp_dir / 'adj.csv')} "
                    f"--gaussian-path {quote(tmp_dir / 'samples.csv')} "
                    f"--output-dir {quote(output_dir)} "
                    f"--enabled-models {shlex.quote(enabled_models)} "
                    f"--causal-llm-backbone {shlex.quote(causal_llm_backbone)}"
                ),
            ),
        ]
    return f"UnifiedOneShot_{problem}", commands


def build_camml_job(problem: str, overrides: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    solver = load_solver_config("LLM_CaMML")
    size = overrides.get("cammml.size", solver_size_for_problem(solver, problem, DEFAULT_CAMML_SIZE))
    repeat = overrides.get("cammml.repeat", solver_value(solver, "repeat", "1"))
    conf = overrides.get("cammml.conf", solver_value(solver, "conf", "0.99999"))
    prior = {"ancs": [], "forb_ancs": []}
    output_dir = method_output_dir("LLM_CaMML", problem)
    output_dir.mkdir(parents=True, exist_ok=True)
    dag_dir = output_dir / "graphs"
    dag_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "results.txt"
    x, y, nodes = _load_processed_bundle(problem)
    csv_name = f"{problem}_{size}_{repeat}.csv"
    _write_bn_assets(
        problem=problem,
        x=x,
        y=y,
        nodes=nodes,
        camml_csv_name=csv_name,
        minobsx_bdeu_name=f"{problem}_{size}_{repeat}.BDeu",
    )

    command = (
        f"cd {quote(CAMML_ROOT)} && "
        f"{quote(PYTHON_EXEC)} {quote(CAMML_ROOT / 'CaMML_perform.py')} "
        f"--d={shlex.quote(problem)} --N={shlex.quote(size)} --r={shlex.quote(repeat)} "
        f"--ancs={shlex.quote(str(prior['ancs']))} --forb_ancs={shlex.quote(str(prior['forb_ancs']))} --abs_edges='[]' "
        f"--output={quote(log_path)} --dag_path={quote(dag_dir)} --conf={shlex.quote(conf)}"
    )
    return f"LLM_CaMML_{problem}", [(stage_label("LLM_CaMML", problem, "方法运行"), command)]


def build_minobsx_job(problem: str, overrides: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    solver = load_solver_config("LLM_MINOBSx")
    size = overrides.get("minobsx.size", solver_size_for_problem(solver, problem, DEFAULT_MINOBSX_SIZE))
    repeat = overrides.get("minobsx.repeat", solver_value(solver, "repeat", "1"))
    timeout = overrides.get("minobsx.timeout", solver_value(solver, "timeout", ""))
    prior = {"ancs": [], "forb_ancs": [], "order": []}
    output_dir = method_output_dir("LLM_MINOBSx", problem)
    output_dir.mkdir(parents=True, exist_ok=True)
    dag_dir = output_dir / "graphs"
    dag_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "results.txt"
    x, y, nodes = _load_processed_bundle(problem)
    csv_name = f"{problem}_{size}_{repeat}.csv"
    _write_bn_assets(
        problem=problem,
        x=x,
        y=y,
        nodes=nodes,
        camml_csv_name=csv_name,
        minobsx_bdeu_name=f"{problem}_{size}_{repeat}.BDeu",
    )

    timeout_prefix = f"timeout {shlex.quote(timeout)} " if timeout else ""
    command = (
        f"cd {quote(MINOBSX_ROOT)} && "
        f"{timeout_prefix}{quote(PYTHON_EXEC)} {quote(MINOBSX_ROOT / 'MINOBSx_perform.py')} "
        f"--d={shlex.quote(problem)} --N={shlex.quote(size)} --r={shlex.quote(repeat)} "
        f"--ancs={shlex.quote(str(prior['ancs']))} --forb_ancs={shlex.quote(str(prior['forb_ancs']))} --order={shlex.quote(str(prior['order']))} --abs_edges='[]' "
        f"--output={quote(log_path)} --dag_path={quote(dag_dir)}"
    )
    return f"LLM_MINOBSx_{problem}", [(stage_label("LLM_MINOBSx", problem, "方法运行"), command)]


def build_ilscsl_job(problem: str, overrides: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    solver = load_solver_config("ILSCSL")
    alg = overrides.get("ilscsl.alg", solver_value(solver, "alg", "HC"))
    score = overrides.get("ilscsl.score", solver_value(solver, "score", "bdeu"))
    model = overrides.get("ilscsl.model", solver_value(solver, "model", "gpt-4"))
    data_index = overrides.get("ilscsl.data_index", solver_value(solver, "data_index", "1"))
    datasize_index = overrides.get("ilscsl.datasize_index", solver_value(solver, "datasize_index", "0"))
    llm_url = overrides.get("ilscsl.llm_url") or os.environ.get("ILS_CSL_LLM_URL", "")
    sizes = ILSCSL_DATASET_SIZES.get(problem)
    if sizes is None:
        size = overrides.get("ilscsl.size", DEFAULT_ILSCSL_SIZE)
    else:
        try:
            size = sizes[int(datasize_index)]
        except (ValueError, IndexError):
            size = sizes[0]

    if alg == "CaMML":
        expected_input = ILSCSL_ROOT / "data" / "csv" / f"{problem}_{size}_{data_index}.csv"
    else:
        expected_input = ILSCSL_ROOT / "data" / "score" / score / f"{problem}_{size}_{data_index}.txt"
    if not expected_input.exists():
        raise FileNotFoundError(
            f"ILS-CSL input not found for {problem} ({alg}): {expected_input}. "
            "The clone currently ships only a subset of the original paper's data."
        )

    output_dir = method_output_dir("ILSCSL", problem, alg)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"{problem}-{size}-{data_index}-{alg}-{score}"
    log_filepath = overrides.get("ilscsl.log_filepath") or str(output_dir / f"{exp_name}.log")
    command = (
        f"cd {quote(ILSCSL_ROOT)} && "
        f"ILS_CSL_LLM_URL={shlex.quote(llm_url)} "
        f"{quote(PYTHON_EXEC)} {quote(ILSCSL_ROOT / 'main.py')} "
        f"--dataset={shlex.quote(problem)} --model={shlex.quote(model)} --alg={shlex.quote(alg)} "
        f"--score={shlex.quote(score)} --data_index={shlex.quote(data_index)} --datasize_index={shlex.quote(datasize_index)} "
        f"--log_filepath={shlex.quote(log_filepath)} "
        f"&& mkdir -p {quote(output_dir)} "
        f"&& cp {quote(ILSCSL_ROOT / 'out' / 'metrics' / f'{exp_name}.csv')} {quote(output_dir / 'metrics.csv')} "
        f"&& cp {quote(ILSCSL_ROOT / 'out' / 'prior-iter' / f'{exp_name}.json')} {quote(output_dir / 'prior_iter.json')}"
    )
    return f"ILSCSL_{problem}", [
        (stage_label("ILSCSL", problem, "方法运行"), command),
        (stage_label("ILSCSL", problem, "保存结果"), f"test -f {quote(output_dir / 'metrics.csv')} && test -f {quote(output_dir / 'prior_iter.json')}"),
    ]


def build_jtgap_job(problem: str, overrides: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    solver = load_solver_config("JunctionTreeGap")
    output_dir = method_output_dir("JunctionTreeGap", problem)
    top_k = overrides.get("jtgap.top_k", solver_value(solver, "top_k", "3"))
    gap_weight = overrides.get("jtgap.gap_weight", solver_value(solver, "gap_weight", "0.5"))
    sparsity_threshold = overrides.get("jtgap.sparsity_threshold", solver_value(solver, "sparsity_threshold", "0.0"))
    edge_threshold = overrides.get("jtgap.edge_threshold", solver_value(solver, "edge_threshold", "0.0"))
    covariance_jitter = overrides.get("jtgap.covariance_jitter", solver_value(solver, "covariance_jitter", "1e-6"))
    gap_metric = overrides.get("jtgap.gap_metric", solver_value(solver, "gap_metric", "sym_kl"))
    clique_potential_model = overrides.get("jtgap.clique_potential_model", solver_value(solver, "clique_potential_model", "gaussian"))
    separator_model = overrides.get("jtgap.separator_model", solver_value(solver, "separator_model", "gaussian"))
    processed_bundle = processed_problem_path(problem, "data.npy")
    command = (
        f"{quote(PYTHON_EXEC)} {quote(JTGAP_ROOT / 'main.py')} "
        f"--data-npy {quote(processed_problem_path(problem, 'data.npy'))} "
        f"--output-dir {quote(output_dir)} "
        f"--top-k {shlex.quote(top_k)} "
        f"--gap-weight {shlex.quote(gap_weight)} "
        f"--sparsity-threshold {shlex.quote(sparsity_threshold)} "
        f"--edge-threshold {shlex.quote(edge_threshold)} "
        f"--covariance-jitter {shlex.quote(covariance_jitter)} "
        f"--gap-metric {shlex.quote(gap_metric)} "
        f"--clique-potential-model {shlex.quote(clique_potential_model)} "
        f"--separator-model {shlex.quote(separator_model)}"
    )
    commands = (
        [(stage_label("JunctionTreeGap", problem, "方法运行"), command)]
        if processed_bundle.exists()
        else [
            (stage_label("JunctionTreeGap", problem, "数据处理"), preprocess_command(problem)),
            (stage_label("JunctionTreeGap", problem, "方法运行"), command),
        ]
    )
    return f"JunctionTreeGap_{problem}", commands


def build_backend_job(
    problem: str, overrides: dict[str, str], backend: str = "CaMML"
) -> tuple[str, list[str]]:
    backend = overrides.get("llmcd.backend", backend)
    if backend == "CaMML":
        return build_camml_job(problem, overrides)
    if backend == "MINOBSx":
        return build_minobsx_job(problem, overrides)
    raise ValueError(f"Unsupported backend: {backend}")


JOB_BUILDERS = {
    "DataAssist": build_dataassist_job,
    "DataAssist_SmBFO": build_dataassist_job,
    "GUIDE": build_guide_job,
    "GUIDE_experiment": build_guide_job,
    "GUIDE_gaussian_30": build_guide_job,
    "GUIDE_gaussian_50": build_guide_job,
    "KCRL": build_kcrl_job,
    "KCRL_experiment": build_kcrl_job,
    "KCRL_asia": build_kcrl_job,
    "KCRL_lucas": build_kcrl_job,
    "KCRL_sachs": build_kcrl_job,
    "KCRL_Oxygen-therapy": build_kcrl_job,
    "UnifiedOneShot": build_unified_job,
    "UnifiedOneShot_experiment": build_unified_job,
    "UnifiedOneShot_asia": build_unified_job,
    "UnifiedOneShot_alarm": build_unified_job,
    "UnifiedOneShot_sachs": build_unified_job,
    "LLM_CaMML": partial(build_backend_job, backend="CaMML"),
    "LLM_CaMML_experiment": partial(build_backend_job, backend="CaMML"),
    "CaMML": partial(build_backend_job, backend="CaMML"),
    "LLM_MINOBSx": partial(build_backend_job, backend="MINOBSx"),
    "LLM_MINOBSx_experiment": partial(build_backend_job, backend="MINOBSx"),
    "MINOBSx": partial(build_backend_job, backend="MINOBSx"),
    "ILSCSL": build_ilscsl_job,
    "ILSCSL_experiment": build_ilscsl_job,
    "JunctionTreeGap": build_jtgap_job,
    "JunctionTreeGap_experiment": build_jtgap_job,
    "JTGap": build_jtgap_job,
    "JTGap_experiment": build_jtgap_job,
}


def resolve_job(overrides: dict[str, str]) -> tuple[str, list[str]]:
    problem = overrides.get("problem")
    if not problem:
        raise ValueError("Missing required override: problem=<dataset>")

    generator = (
        overrides.get("problem.generator")
        or overrides.get("generator")
        or overrides.get("experiment")
    )
    if not generator:
        raise ValueError(
            "Missing required override: problem.generator=<DataAssist|GUIDE|KCRL|UnifiedOneShot|LLM_CaMML|LLM_MINOBSx|ILSCSL>"
        )

    normalized = generator.strip()
    builder = JOB_BUILDERS.get(normalized)
    if builder is None:
        raise ValueError(f"Unsupported generator/experiment: {normalized}")
    return builder(problem, overrides)


def run_command_block(
    *,
    config_path: Path,
    commands: list[tuple[str, str]],
    step: str,
    dry_run: bool,
    label: str,
) -> int:
    selected_commands = select_commands(commands, step)
    print(f"Experiment: {label}")
    print(f"Config: {config_path}")
    print(f"Workdir: {PROJECT_ROOT}")
    for index, stage_label_text, command in selected_commands:
        print(f"{stage_label_text} {command}")
        if dry_run:
            continue
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        result = subprocess.run(command, shell=True, cwd=str(PROJECT_ROOT), env=env, check=False)
        if result.returncode != 0:
            print(
                f"{stage_label_text} failed with exit code {result.returncode}",
                file=sys.stderr,
            )
            return result.returncode
    return 0


def main() -> int:
    args = parse_args()
    config_path = (Path(__file__).resolve().parent / f"{args.config_name}.yaml").resolve()
    load_config(config_path)
    override_grid = expand_override_grid(parse_overrides(args.overrides))
    failures = 0
    for overrides in override_grid:
        label, commands = resolve_job(overrides)
        result = run_command_block(config_path=config_path, commands=commands, step=args.step, dry_run=args.dry_run, label=label)
        if result != 0:
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
