import argparse
import os
from pathlib import Path

import pandas as pd

from data_loader import load_data
from models import initialize_models, train_models
from evaluation import evaluate_and_save_results


def main(
    ground_truth_path,
    gaussian_path=None,
    non_gaussian_path=None,
    extra_adj_matrix_path=None,
    output_dir=None,
    enabled_models=None,
    causal_llm_backbone=None,
):
    ground_truth_name = os.path.splitext(os.path.basename(ground_truth_path))[0]
    ground_truth, gaussian_samples, non_gaussian_samples, node_labels = load_data(
        ground_truth_path, gaussian_path, non_gaussian_path
    )

    extra_adj_matrix = None
    if extra_adj_matrix_path:
        extra_adj_matrix = pd.read_csv(extra_adj_matrix_path, header=None).values

    combined_results = {}
    use_suffix = gaussian_samples is not None and non_gaussian_samples is not None
    original_cwd = os.getcwd()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)

    try:
        if gaussian_samples is not None:
            models_g = initialize_models(
                gaussian_samples.shape[1],
                enabled_models=enabled_models,
                causal_llm_backbone=causal_llm_backbone,
            )
            combined_results['g'] = train_models(
                models_g, gaussian_samples, 'g', node_labels, ground_truth_name
            )

        if non_gaussian_samples is not None:
            models_ng = initialize_models(
                non_gaussian_samples.shape[1],
                enabled_models=enabled_models,
                causal_llm_backbone=causal_llm_backbone,
            )
            combined_results['ng'] = train_models(
                models_ng, non_gaussian_samples, 'ng', node_labels, ground_truth_name
            )

        evaluate_and_save_results(
            ground_truth, combined_results, ground_truth_name, extra_adj_matrix, use_suffix
        )
    finally:
        if output_dir:
            os.chdir(original_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Unified One-Shot baseline.")
    parser.add_argument("--ground-truth-path", required=True)
    parser.add_argument("--gaussian-path", default=None)
    parser.add_argument("--non-gaussian-path", default=None)
    parser.add_argument("--extra-adj-matrix-path", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--enabled-models",
        default="PC",
        help="Comma-separated model names, e.g. PC,GES or PC only.",
    )
    parser.add_argument(
        "--causal-llm-backbone",
        default="Llama",
        help="Backbone for the causal_llm model: Llama, GPT2, GPTNeoX, Gemma, DeepseekV3.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enabled_models = [item.strip() for item in args.enabled_models.split(",") if item.strip()]
    main(
        ground_truth_path=args.ground_truth_path,
        gaussian_path=args.gaussian_path,
        non_gaussian_path=args.non_gaussian_path,
        extra_adj_matrix_path=args.extra_adj_matrix_path,
        output_dir=args.output_dir,
        enabled_models=enabled_models,
        causal_llm_backbone=args.causal_llm_backbone,
    )

