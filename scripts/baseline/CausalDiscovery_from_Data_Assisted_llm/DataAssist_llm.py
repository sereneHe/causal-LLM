#!/usr/bin/env python3
"""Generate LLM-assisted prior knowledge for SmBFO causal discovery.

This script extracts the pairwise LLM prior-generation logic from
`causal_llm_1.ipynb` and writes reusable prior artifacts for downstream
constrained DAG learning.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from castle.common.priori_knowledge import PrioriKnowledge
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI


ALL_VARS = {
    "Alkali_Cations": 0,
    "Transition_Metal_Cations": 1,
    "Lattice_Parameter": 2,
    "Composition": 3,
    "Unit_Cell_Angle": 4,
    "Volume": 5,
    "In_Plane_Polarization": 6,
}


def get_llm_info(llm: ChatOpenAI, agent, var_1: str, var_2: str) -> tuple[str, str]:
    retrieval = agent(
        f"Does {var_1} cause {var_2} or the other way around? "
        "We assume the following definition of causation: if we change A, B will also change. "
        "The relationship does not have to be linear or monotonic. We are interested in all "
        "types of causal relationships, including partial and indirect relationships, given "
        "that our definition holds."
    )
    raw_output = retrieval["output"] if isinstance(retrieval, dict) else str(retrieval)
    prediction = llm.predict(
        "We assume the following definition of causation: if we change A, B will also change. "
        f"Based on the following information: {raw_output}, print (0,1) if {var_1} causes {var_2}, "
        f"print (1, 0) if {var_2} causes {var_1}, print (0,0) if there is no causal relationship "
        "between the variables. Finally, print (-1, -1) if you don't know. "
        "Importantly, don't try to make up an answer if you don't know."
    ).strip()
    return raw_output, prediction


def build_prior(model_name: str) -> tuple[PrioriKnowledge, list[dict], list[list[int]], list[list[int]]]:
    llm = ChatOpenAI(temperature=0, model=model_name)
    tools = load_tools(["arxiv"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
    )

    prior = PrioriKnowledge(n_nodes=len(ALL_VARS))
    records: list[dict] = []
    required_edges: list[list[int]] = []
    forbidden_edges: list[list[int]] = []

    for var_1, var_2 in combinations(ALL_VARS.keys(), r=2):
        evidence, prediction = get_llm_info(llm, agent, var_1, var_2)
        record = {
            "var_1": var_1,
            "var_2": var_2,
            "prediction": prediction,
            "evidence": evidence,
        }

        if prediction == "(0,1)":
            prior.add_required_edges([(ALL_VARS[var_1], ALL_VARS[var_2])])
            prior.add_forbidden_edges([(ALL_VARS[var_2], ALL_VARS[var_1])])
            required_edge = [ALL_VARS[var_1], ALL_VARS[var_2]]
            forbidden_edge = [ALL_VARS[var_2], ALL_VARS[var_1]]
            record["required_edge"] = required_edge
            record["forbidden_edge"] = forbidden_edge
            required_edges.append(required_edge)
            forbidden_edges.append(forbidden_edge)
        elif prediction == "(1,0)":
            prior.add_required_edges([(ALL_VARS[var_2], ALL_VARS[var_1])])
            prior.add_forbidden_edges([(ALL_VARS[var_1], ALL_VARS[var_2])])
            required_edge = [ALL_VARS[var_2], ALL_VARS[var_1]]
            forbidden_edge = [ALL_VARS[var_1], ALL_VARS[var_2]]
            record["required_edge"] = required_edge
            record["forbidden_edge"] = forbidden_edge
            required_edges.append(required_edge)
            forbidden_edges.append(forbidden_edge)
        else:
            record["required_edge"] = None
            record["forbidden_edge"] = None

        records.append(record)

    return prior, records, required_edges, forbidden_edges


def named_matrix(matrix: np.ndarray) -> pd.DataFrame:
    inverse_var_map = {v: k for k, v in ALL_VARS.items()}
    labels = [inverse_var_map[i] for i in range(matrix.shape[0])]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM prior knowledge for SmBFO variables.")
    parser.add_argument(
        "--model",
        default="gpt-4-turbo",
        help="Chat model name passed to LangChain ChatOpenAI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where prior artifacts will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prior, records, required_edges, forbidden_edges = build_prior(args.model)
    prior_matrix = np.clip(prior.matrix, 0, 1)
    named_matrix(prior_matrix).to_csv(args.output_dir / "llm_prior_matrix.csv")
    pd.DataFrame(records).to_csv(args.output_dir / "llm_pairwise_queries.csv", index=False)

    payload = {
        "variables": ALL_VARS,
        "required_edges": required_edges,
        "forbidden_edges": forbidden_edges,
    }
    (args.output_dir / "llm_prior_edges.json").write_text(json.dumps(payload, indent=2))

    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
