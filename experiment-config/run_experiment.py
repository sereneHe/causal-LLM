#!/usr/bin/env python3
"""Run experiment commands defined in a Hydra-style top-level config."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from itertools import product

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute commands from a Hydra-style top-level configuration file."
    )
    parser.add_argument(
        "--multirun",
        action="store_true",
        help="Run one or more experiments defined in a top-level config file.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Top-level config filename without .yaml, used with --multirun.",
    )
    parser.add_argument(
        "--step",
        default="all",
        help="Step number to run (1-based) or 'all'. Defaults to all.",
    )
    parser.add_argument(
        "--solver-config",
        default=None,
        help="Optional absolute path to a solver YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra-style key=value overrides for --multirun.",
    )
    args = parser.parse_args()
    if not args.multirun:
        parser.error("Only --multirun mode is supported.")
    if not args.config_name:
        parser.error("--config-name is required when --multirun is used.")
    return args


def load_problem_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return config


def load_yaml_config(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return config


def select_commands(commands: list[str], step: str) -> list[tuple[int, str]]:
    if step == "all":
        return list(enumerate(commands, start=1))

    try:
        index = int(step)
    except ValueError as exc:
        raise ValueError("--step must be 'all' or a 1-based integer") from exc

    if index < 1 or index > len(commands):
        raise ValueError(f"--step must be between 1 and {len(commands)}")

    return [(index, commands[index - 1])]


def parse_overrides(items: list[str]) -> dict[str, list[str]]:
    overrides: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")
        values = [part.strip().strip("'").strip('"') for part in value.split(",") if part.strip()]
        overrides[key] = values or [""]
    return overrides


def expand_override_grid(overrides: dict[str, list[str]]) -> list[dict[str, str]]:
    if not overrides:
        return [{}]
    keys = list(overrides)
    values_product = product(*(overrides[key] for key in keys))
    return [dict(zip(keys, combo)) for combo in values_product]


def resolve_experiment_name(config: dict, overrides: dict[str, str]) -> str:
    experiment_name = overrides.get("experiment")
    if experiment_name:
        return experiment_name
    problem_name = overrides.get("problem")
    generator_name = overrides.get("problem.generator") or overrides.get("generator")
    if problem_name and generator_name:
        return f"{generator_name}_{problem_name}"
    config_default = config.get("experiment")
    if isinstance(config_default, str) and config_default.strip():
        return config_default.strip()
    raise ValueError("Could not resolve experiment name from overrides or config.")


def resolve_multirun_job(config_path: Path, config: dict, overrides: dict[str, str]) -> tuple[str, dict]:
    experiments = config.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError(f"No experiments mapping found in config: {config_path}")
    experiment_name = resolve_experiment_name(config, overrides)
    job = experiments.get(experiment_name)
    if not isinstance(job, dict):
        available = ", ".join(sorted(experiments))
        raise ValueError(
            f"Experiment '{experiment_name}' not found in {config_path}. Available: {available}"
        )
    merged_job = dict(job)
    merged_paths = {}
    base_paths = config.get("paths")
    if isinstance(base_paths, dict):
        merged_paths.update(base_paths)
    job_paths = job.get("paths")
    if isinstance(job_paths, dict):
        merged_paths.update(job_paths)
    if merged_paths:
        merged_job["paths"] = merged_paths
    return experiment_name, merged_job


def run_command_block(
    *,
    config_path: Path,
    workdir: Path,
    commands: list[str],
    step: str,
    dry_run: bool,
    solver_config_path: Path | None,
    solver_config: dict | None,
    label: str | None = None,
) -> int:
    if not isinstance(commands, list) or not commands:
        raise ValueError(f"No commands found in config: {config_path}")

    selected_commands = select_commands(commands, step)
    if label:
        print(f"Experiment: {label}")
    print(f"Problem config: {config_path}")
    print(f"Workdir: {workdir}")
    if solver_config_path is not None:
        print(f"Solver config: {solver_config_path}")

    for index, command in selected_commands:
        print(f"[step {index}] {command}")
        if dry_run:
            continue

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        if solver_config is not None:
            env["SOLVER_CONFIG_PATH"] = str(solver_config_path)
            for key, value in solver_config.items():
                env_key = f"SOLVER_{key}".upper()
                if isinstance(value, (list, tuple)):
                    env[env_key] = ",".join(str(item) for item in value)
                else:
                    env[env_key] = str(value)
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            env=env,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"Step {index} failed with exit code {result.returncode}",
                file=sys.stderr,
            )
            return result.returncode
    return 0


def main() -> int:
    args = parse_args()
    solver_config_path = (
        Path(args.solver_config).expanduser().resolve()
        if args.solver_config is not None
        else None
    )
    solver_config = load_yaml_config(solver_config_path)
    config_path = (
        Path(__file__).resolve().parent / f"{args.config_name}.yaml"
    ).resolve()
    config = load_problem_config(config_path)
    override_grid = expand_override_grid(parse_overrides(args.overrides))
    for overrides in override_grid:
        experiment_name, job = resolve_multirun_job(config_path, config, overrides)
        commands = job.get("commands", [])
        workdir = Path(job.get("paths", {}).get("workdir", config_path.parent)).resolve()
        result = run_command_block(
            config_path=config_path,
            workdir=workdir,
            commands=commands,
            step=args.step,
            dry_run=args.dry_run,
            solver_config_path=solver_config_path,
            solver_config=solver_config,
            label=experiment_name,
        )
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
