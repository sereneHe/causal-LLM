from __future__ import annotations

"""Codiet data staging and standardization utilities.

Codiet is handled here as a two-stage source entirely inside this repo:

* primary path: the copied ``dataset/raw/codiet`` bundle, which provides the
  merged observational table and the knowledge graph intersection
* fallback path: the original source-repo style graphml + IPC/feather inputs

The preprocessing entry point stays thin and only selects which bundle to
materialize.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/xiaoyuhe/Causal-LLM")
CODIET_DATA_ROOT = PROJECT_ROOT / "dataset/raw/codiet"


def _normalize_name(name: str) -> str:
    normalized = str(name).strip().lower()
    normalized = normalized.replace("%", " pct ")
    normalized = normalized.replace("(", " ").replace(")", " ")
    normalized = normalized.replace("[", " ").replace("]", " ")
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _rec_data(*parts: str) -> Path:
    return CODIET_DATA_ROOT.joinpath(*parts)


def _calculate_reverse_score(values, lower_bound, upper_bound, max_points):
    scores = max_points - (max_points * (values - lower_bound) / (upper_bound - lower_bound))
    return scores.clip(lower=0, upper=max_points)


def _calculate_adequacy_components(dat):
    adequacy_components = [
        ("heiveg", "lvtotal", 0, 1.1, 5),
        ("heibngrn", "lbeangrn", 0, 0.2, 5),
        ("heitotfrt", "T_F_TOTAL", 0, 0.8, 5),
        ("heiwholefrt", "WHOLEFRT", 0, 0.4, 5),
        ("heiwholegrain", "T_G_WHOLE", 0, 1.5, 10),
        ("heidairy", "T_D_TOTAL", 0, 1.3, 10),
        ("heitotpro", "lallmeat", 0, 2.5, 5),
        ("heiseaplantpro", "lseaplant", 0, 0.8, 5),
        ("heifattyacid", "faratio", 1.2, 2.5, 10),
    ]
    for component, numerator, min_value, max_value, max_points in adequacy_components:
        if numerator == "faratio":
            density = np.where(dat["TSFAT"] > 0, dat["MONOPOLY"] / dat["TSFAT"], max_value)
        else:
            density = dat[numerator] / (dat["TKCAL"] / 1000)
        dat[component] = max_points * (density - min_value) / (max_value - min_value)
        dat[component] = dat[component].clip(lower=0, upper=max_points)


def _calculate_moderation_components(dat):
    dat["sodden"] = dat["TSODI"] / dat["TKCAL"]
    dat["heisodi"] = _calculate_reverse_score(dat["sodden"], 1.1, 2.0, 10)
    dat["refgrainnden"] = dat["T_G_REFINED"] / (dat["TKCAL"] / 1000)
    dat["heirefgrain"] = _calculate_reverse_score(dat["refgrainnden"], 1.8, 4.3, 10)
    dat["sofa_perc"] = 100 * (dat["EMPTYCAL10"] / dat["TKCAL"])
    dat["heisofaas"] = _calculate_reverse_score(dat["sofa_perc"], 19, 50, 20)


def _leg_all(dat: pd.DataFrame) -> pd.DataFrame:
    dat["mbmax"] = 2.5 * (dat["TKCAL"] / 1000)
    dat["meatleg"] = np.where(dat["ALLMEAT"] < dat["mbmax"], dat["T_V_LEGUMES"] * 4, 0)
    dat["needmeat"] = np.where(dat["ALLMEAT"] < dat["mbmax"], dat["mbmax"] - dat["ALLMEAT"], 0)
    dat["lallmeat"] = np.where(dat["meatleg"] <= dat["needmeat"], dat["ALLMEAT"] + dat["meatleg"], 0)
    dat["lseaplant"] = np.where(dat["meatleg"] <= dat["needmeat"], dat["SEAPLANT"] + dat["meatleg"], 0)
    dat["lvtotal"] = np.where(dat["meatleg"] <= dat["needmeat"], dat["T_V_TOTAL"], 0)
    dat["lbeangrn"] = np.where(dat["meatleg"] <= dat["needmeat"], dat["T_V_DRKGR"], 0)
    dat["extrameat"] = np.where(dat["meatleg"] > dat["needmeat"], dat["meatleg"] - dat["needmeat"], 0)
    dat["extraleg"] = np.where(dat["meatleg"] > dat["needmeat"], dat["extrameat"] / 4, 0)
    dat["lallmeat"] = np.where(dat["meatleg"] > dat["needmeat"], dat["ALLMEAT"] + dat["needmeat"], dat["lallmeat"])
    dat["lseaplant"] = np.where(dat["meatleg"] > dat["needmeat"], dat["SEAPLANT"] + dat["needmeat"], dat["lseaplant"])
    dat["lvtotal"] = np.where(dat["meatleg"] > dat["needmeat"], dat["T_V_TOTAL"] + dat["extraleg"], dat["lvtotal"])
    dat["lbeangrn"] = np.where(dat["meatleg"] > dat["needmeat"], dat["T_V_DRKGR"] + dat["extraleg"], dat["lbeangrn"])
    dat["lallmeat"] = np.where(dat["ALLMEAT"] >= dat["mbmax"], dat["ALLMEAT"], dat["lallmeat"])
    dat["lseaplant"] = np.where(dat["ALLMEAT"] >= dat["mbmax"], dat["SEAPLANT"], dat["lseaplant"])
    dat["lvtotal"] = np.where(dat["ALLMEAT"] >= dat["mbmax"], dat["T_V_TOTAL"] + dat["T_V_LEGUMES"], dat["lvtotal"])
    dat["lbeangrn"] = np.where(dat["ALLMEAT"] >= dat["mbmax"], dat["T_V_DRKGR"] + dat["T_V_LEGUMES"], dat["lbeangrn"])
    return dat


def _combo(fped: pd.DataFrame, diet: pd.DataFrame, demo: pd.DataFrame | None = None, days: list[int] | None = None, agethresh: int | None = None) -> pd.DataFrame:
    dat = pd.merge(fped, diet, on=["SEQN", "DRSTZ"], how="left")
    if demo is not None:
        dat = pd.merge(dat, demo, on="SEQN", how="left")
    dat = dat[dat["TKCAL"].between(600, 5000)]
    if days is not None:
        dat = dat[dat["DRSTZ"].isin(days)]
    if agethresh:
        dat = dat[dat["RIDAGEYR"] >= agethresh]
    dat["WHOLEFRT"] = dat["T_F_CITMLB"] + dat["T_F_OTHER"]
    dat["MONOPOLY"] = dat["TMFAT"] + dat["TPFAT"]
    dat["ALLMEAT"] = dat["T_PF_MPS_TOTAL"] + dat["T_PF_EGGS"] + dat["T_PF_NUTSDS"] + dat["T_PF_SOY"]
    dat["SEAPLANT"] = dat["T_PF_SEAFD_HI"] + dat["T_PF_SEAFD_LOW"] + dat["T_PF_NUTSDS"] + dat["T_PF_SOY"]
    dat["ADDSUGC"] = 16 * dat["T_ADD_SUGARS"]
    dat["SOLFATC"] = 9 * dat["T_SOLID_FATS"]
    dat["MAXALCGR"] = 13 * (dat["TKCAL"] / 1000)
    dat["EXALCCAL"] = np.where(dat["TALCO"] <= dat["MAXALCGR"], 0, 7 * (dat["TALCO"] - dat["MAXALCGR"]))
    dat["EMPTYCAL10"] = dat["ADDSUGC"] + dat["SOLFATC"] + dat["EXALCCAL"]
    return dat


def _hei(fped: pd.DataFrame, diet: pd.DataFrame, demo: pd.DataFrame | None = None, days: list[int] | None = None, agethresh: int | None = None, return_full_feats: bool = False) -> pd.DataFrame:
    dat = _combo(fped, diet, demo, days, agethresh)
    dat = _leg_all(dat)
    _calculate_adequacy_components(dat)
    _calculate_moderation_components(dat)
    component_columns = [
        "heiveg",
        "heibngrn",
        "heitotfrt",
        "heiwholefrt",
        "heiwholegrain",
        "heidairy",
        "heitotpro",
        "heiseaplantpro",
        "heifattyacid",
        "heirefgrain",
        "heisofaas",
        "heisodi",
    ]
    dat["HEI"] = dat[component_columns].sum(axis=1)
    if return_full_feats:
        return dat
    return dat[["SEQN", "DRSTZ", "RIDAGEYR", "TKCAL", "HEI"] + component_columns]


def _read_hei() -> pd.DataFrame:
    cached = _rec_data("HEI", "hei.pkl")
    if cached.exists():
        try:
            return pd.read_pickle(cached)
        except Exception:
            cached.unlink(missing_ok=True)
    fped = pd.read_csv(_rec_data("HEI", "processed_fped.csv"))
    diet = pd.read_csv(_rec_data("HEI", "processed_diet.csv"))
    hei_data = _hei(fped, diet, None, agethresh=2, return_full_feats=True)
    hei_data.drop(columns=["DRSTZ"], inplace=True)
    missing_v_seqn = [seqn for seqn in hei_data["SEQN"] if "V" not in seqn]
    hei_data.loc[hei_data["SEQN"].isin(missing_v_seqn), "SEQN"] += "_V1"
    hei_data.insert(1, "ID", hei_data["SEQN"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    hei_data.insert(2, "VISIT", hei_data["SEQN"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
    hei_data.set_index(["ID", "VISIT"], inplace=True)
    hei_data.sort_index(inplace=True)
    hei_data.drop(columns=["SEQN"], inplace=True)
    hei_data = hei_data.groupby(level=["ID", "VISIT"]).mean().reset_index()
    food_feats_orig = [
        "TKCAL",
        "WHOLEFRT",
        "MONOPOLY",
        "ALLMEAT",
        "SEAPLANT",
        "ADDSUGC",
        "SOLFATC",
        "TALCO",
        "T_F_TOTAL",
        "T_G_WHOLE",
        "T_D_TOTAL",
        "TSFAT",
        "TSODI",
        "T_G_REFINED",
        "EMPTYCAL10",
        "T_V_TOTAL",
        "T_V_DRKGR",
        "T_V_LEGUMES",
    ]
    hei_data = hei_data[food_feats_orig + ["ID", "VISIT"]]
    rename_dict = {n: "food_" + n for n in food_feats_orig}
    hei_data = hei_data.rename(columns=rename_dict)
    hei_data.to_pickle(cached)
    return hei_data


def _read_body_comp() -> pd.DataFrame:
    cached = _rec_data("body_composition", "body_comp.pkl")
    if cached.exists():
        try:
            return pd.read_pickle(cached)
        except Exception:
            cached.unlink(missing_ok=True)
    body_comp = pd.read_excel(_rec_data("body_composition", "BiosensorsMicrocaya_data_combined_jan2025.xlsx"))
    body_comp.dropna(how="all", inplace=True)
    body_comp.reset_index(drop=True, inplace=True)
    body_comp.columns = body_comp.columns.str.strip()
    body_comp.columns = body_comp.columns.str.replace(" ", "")
    body_comp.insert(1, "ID", body_comp["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    body_comp.insert(2, "VISIT", body_comp["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
    limit_columns = [col for col in body_comp.columns if "limit" in col.lower()]
    single_value_columns = [col for col in body_comp.columns if body_comp[col].nunique() == 1]
    to_drop = ["sample_id", "volunteer_id", "date_of_birth", "exam_date", "recruitment_site"]
    body_comp.drop(columns=limit_columns + single_value_columns + to_drop, inplace=True)
    body_comp.to_pickle(cached)
    body_comp["gender_numeric"] = body_comp["gender"].apply(lambda x: 0 if x == "Male" else 1)
    return body_comp


def _read_blood_data() -> pd.DataFrame:
    cached = _rec_data("UpdatedDataFromSara", "blood_data.pkl")
    if cached.exists():
        try:
            return pd.read_pickle(cached)
        except Exception:
            cached.unlink(missing_ok=True)
    file = _rec_data("UpdatedDataFromSara", "biochemical data all converted values.xlsx")
    data = pd.read_excel(file)
    data.insert(1, "ID", data["ΙD participant / Compound "].apply(lambda x: int(re.findall(r"\d+", str(x))[0])))
    data.rename(columns={"Timepoint": "VISIT"}, inplace=True)
    data["site_numeric"] = data["Site Collection"].map({"AUTH": 1, "BILBAO": 2, "CORK": 3, "ICL": 4, "UVEG": 5})
    blood_data = data.drop(columns=["Site Collection", "ΙD participant / Compound "]).reset_index(drop=True)
    blood_data = blood_data.apply(pd.to_numeric, errors="coerce")
    blood_data.to_pickle(cached)
    return blood_data


def _read_average_expenditure() -> pd.DataFrame:
    cached = _rec_data("energy_expenditure", "average_expenditure.pkl")
    if cached.exists():
        try:
            return pd.read_pickle(cached)
        except Exception:
            cached.unlink(missing_ok=True)
    energy_expenditure = pd.DataFrame()
    for file in sorted((_rec_data("energy_expenditure")).iterdir()):
        if file.suffix.lower() == ".csv":
            tee = pd.read_csv(file)
            tee = tee.dropna(subset=["timepoint"])
            tee = tee[tee["sample_id"].apply(lambda x: len(re.findall(r"\d+", x)) == 2)]
            tee.insert(1, "ID", tee["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
            tee.insert(2, "VISIT", tee["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
            tee.drop(columns=["sample_id"], inplace=True)
            tee.reset_index(drop=True, inplace=True)
            energy_expenditure = tee if energy_expenditure.empty else pd.concat([energy_expenditure, tee], ignore_index=True)
    energy_expenditure.rename(columns={"TEE2": "TEE", "TEE": "TEE_orig"}, inplace=True)
    average_expenditure = energy_expenditure.groupby(["ID", "VISIT"]).agg({"TEE": "mean"}).reset_index()
    average_expenditure.to_pickle(cached)
    energy_expenditure.to_pickle(_rec_data("energy_expenditure", "expenditure.pkl"))
    return average_expenditure


def _load_codiet_local_bundle() -> tuple[list[str], list[str], pd.DataFrame]:
    hei_data = _read_hei()
    blood_data = _read_blood_data()
    body_comp = _read_body_comp()
    average_expenditure = _read_average_expenditure()
    gmwi_data = pd.read_csv(_rec_data("gmwi_data.csv"))
    gmwi_data = gmwi_data[gmwi_data["sample_modified"].apply(lambda x: len(re.findall(r"\d+", x)) > 0)]
    gmwi_data.insert(1, "ID", gmwi_data["sample_modified"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    gmwi_data.insert(2, "VISIT", gmwi_data["sample_modified"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
    gmwi_data.drop(columns=["sample_original", "sample_modified", "site", "HealthStatus", "Visit", "ParticipantID"], inplace=True)
    data = pd.read_excel(_rec_data("UpdatedDataFromSara", "Blood pressure values all sites WP2.xlsx"))
    data.insert(1, "ID", data["Participant  "].apply(lambda x: int(re.findall(r"\d+", str(x))[0])))
    blood_pressure = data.drop(columns=["Site Collection", "hypertension/medication", "Sex", "Participant  ", "Age"])
    microbiome_alpha_df = pd.read_csv(_rec_data("microbiome", "alpha_summary_CoDiet_total_v2.csv"))
    microbiome_alpha_df = microbiome_alpha_df[microbiome_alpha_df["Unnamed: 0"].str.contains("CD_", na=False)]
    microbiome_alpha_df.insert(1, "ID", microbiome_alpha_df["Unnamed: 0"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    microbiome_alpha_df.insert(2, "VISIT", microbiome_alpha_df["Unnamed: 0"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
    microbiome_alpha_df = microbiome_alpha_df.drop(columns=["Unnamed: 0", "Unnamed: 4", "Unnamed: 5"])
    microbiome_alpha_df = microbiome_alpha_df.rename(columns={c: "microbiome_" + c for c in microbiome_alpha_df.columns if c not in ["ID", "VISIT"]})

    def cleaned_loader(fname, feat_name, reader_func=pd.read_csv):
        df = reader_func(fname)
        df.insert(0, "ID", df["patient"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
        df.insert(1, "VISIT", df["visit"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
        df = df.drop(columns=["patient", "visit"])
        df = df.rename(columns={c: feat_name + "_" + c for c in df.columns if c not in ["ID", "VISIT"]})
        cols_with_many_nans = list(df.columns[df.isnull().sum() > 10])
        return df.drop(columns=cols_with_many_nans)

    scafs_df = cleaned_loader(_rec_data("more_biomarkers", "scafs-stool.csv"), "scafs")
    scafs_df = scafs_df[["ID", "VISIT", "scafs_acetate", "scafs_butyrate", "scafs_formate", "scafs_propionate"]]
    ms_urine_df = cleaned_loader(_rec_data("more_biomarkers", "ms-urine.csv"), "ms_urine")
    ms_urine_df = ms_urine_df.drop(columns=[c for c in ["ms_urine_type", "ms_urine_sample-type"] if c in ms_urine_df.columns])
    ms_serum_df = cleaned_loader(_rec_data("more_biomarkers", "ms-serum.csv"), "ms_serum")
    ms_serum_df = ms_serum_df.drop(columns=[c for c in ["ms_serum_type", "ms_serum_sample-type"] if c in ms_serum_df.columns])
    nmr_urine_df = cleaned_loader(_rec_data("UpdatedNMRLipids_12_25", "unified-nmr-targeted-urine_v2.xlsx"), "nmr_urine", reader_func=pd.read_excel)
    nmr_urine_df.drop(columns=[c for c in ["nmr_urine_site", "nmr_urine_1-methyladenosine"] if c in nmr_urine_df.columns], inplace=True)
    df = pd.read_excel(_rec_data("lipidomics", "lipidomics.xlsx"))
    df = df[df["type"] == "sample"].drop(columns=["type"])
    lipidomics_df = cleaned_loader("", "ms_lip", reader_func=lambda fn: df)
    lipidomics_dbs_rbc_df = cleaned_loader(_rec_data("lipidomics", "lipidomics-dbs-rbc.xlsx"), "dbs_rbc_lip", reader_func=pd.read_excel)
    microbiome_cl_df = pd.read_csv(_rec_data("derived", "microbiome_4_clusters.csv")).drop(columns=["Unnamed: 0"])
    microbiome_phyl_cl_df = pd.read_csv(_rec_data("derived", "microbiome_phylumn4_clusters.csv")).drop(columns=["Unnamed: 0"])
    microbiome_embedding_df = pd.read_csv(_rec_data("derived", "microbiome_embedding_20.csv"))
    microbiome_clean15_df = pd.read_csv(_rec_data("derived", "microbiome_clean15.csv"))

    df_pairs = [
        ("microbiome", microbiome_alpha_df),
        ("scafs", scafs_df),
        ("ms_serum", ms_serum_df),
        ("ms_urine", ms_urine_df),
        ("nmr_urine", nmr_urine_df),
        ("lipidomics", lipidomics_df),
        ("lipidomics_dbs_rbc", lipidomics_dbs_rbc_df),
        ("microbiome_4_cl", microbiome_cl_df),
        ("microbiome_phyl4_cl", microbiome_phyl_cl_df),
        ("microbiome_embedding", microbiome_embedding_df),
        ("microbiome_clean15", microbiome_clean15_df),
    ]
    new_dfs = dict(df_pairs)
    new_dfs_names = [p[0] for p in df_pairs]

    def average_visits(df):
        df = df.groupby("ID").mean(numeric_only=True)
        if "VISIT" in df.columns:
            df.drop(columns=["VISIT"], inplace=True)
        df.reset_index(inplace=True)
        return df

    from functools import reduce

    prep_hei_data = average_visits(hei_data)
    prep_average_expenditure = average_visits(average_expenditure)
    prep_body_comp_data = average_visits(body_comp.select_dtypes(include=["number"]))
    prep_blood_data = average_visits(blood_data.select_dtypes(include=["number"]))
    prep_gmwi_data = average_visits(gmwi_data)
    new_dfs_prep = {}
    for name in new_dfs_names:
        df = average_visits(new_dfs[name])
        new_dfs_prep[name] = df

    prep_data = reduce(
        lambda left, right: pd.merge(left, right, on="ID", how="inner"),
        [prep_hei_data, prep_average_expenditure, prep_body_comp_data, prep_blood_data, prep_gmwi_data, blood_pressure]
        + list(new_dfs_prep.values()),
    )
    prep_data["site_continental"] = prep_data["site_numeric"].isin(
        [1, 2, 5]
    ).astype(int)
    prep_data["normed_TEE"] = prep_data["TEE"] / prep_data["lean_mass_of_trunk"]
    for c in [c for c in prep_data.columns if c.startswith("food_")]:
        prep_data["normed_" + c] = prep_data[c] / prep_data["lean_mass_of_trunk"]

    food_feats = [c for c in prep_data if c.startswith("normed_food_")] + ["food_TKCAL"]
    non_food_feats = ["normed_TEE", "GMWI"]
    for name in ["microbiome", "scafs", "ms_serum.pca", "ms_urine.pca", "nmr_urine", "nmr_urine.pca"]:
        non_food_feats += [c for c in prep_data.columns if name + "_" in c]
    non_food_feats += [c for c in prep_data.columns if "ms_lip_" in c]
    non_food_feats += [c for c in prep_data.columns if "dbs_rbc_lip_" in c]
    return food_feats, non_food_feats, prep_data


def _save_bundle(
    processed_dir: Path,
    x,
    y=None,
    nodes=None,
    csv_name: str | None = None,
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    x_np = np.asarray(x)
    y_np = None if y is None else np.asarray(y)
    if y_np is not None:
        if y_np.ndim != 2 or y_np.shape[0] != y_np.shape[1]:
            raise ValueError(f"Expected a square adjacency matrix, got {y_np.shape}")
        y_np = (np.abs(y_np) > 1e-12).astype(int)
        np.fill_diagonal(y_np, 0)
    if csv_name is not None:
        pd.DataFrame(x_np).to_csv(processed_dir / csv_name, index=False)
    np.save(processed_dir / "X.npy", x_np)
    if y_np is not None:
        np.save(processed_dir / "adj.npy", y_np)
        np.save(processed_dir / "DAG.npy", y_np)
    if nodes is not None:
        np.save(processed_dir / "nodes.npy", np.asarray(nodes))
    bundle_path = processed_dir / "data.npy"
    np.save(bundle_path, {"x": x_np, "y": y_np}, allow_pickle=True)
    return bundle_path


def _read_arrow_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".feather", ".ipc", ".arrow"}:
        try:
            import polars as pl

            if suffix == ".ipc":
                return pl.read_ipc(path).to_pandas()
            return pl.read_ipc(path).to_pandas() if suffix == ".arrow" else pd.read_feather(path)
        except Exception:
            return pd.read_feather(path)
    return pd.read_feather(path)


def _resolve_requested_features(requested: Iterable[str] | None, columns: Iterable[str]) -> list[str]:
    columns = list(columns)
    if requested is None:
        return columns

    normalized_lookup: dict[str, str] = {}
    for column in columns:
        normalized_lookup.setdefault(_normalize_name(column), column)

    resolved: list[str] = []
    missing: list[str] = []
    for feature in requested:
        if feature in columns:
            resolved.append(feature)
            continue
        normalized = _normalize_name(feature)
        if normalized in normalized_lookup:
            resolved.append(normalized_lookup[normalized])
            continue
        fuzzy = [column for column in columns if normalized and normalized in _normalize_name(column)]
        if len(fuzzy) == 1:
            resolved.append(fuzzy[0])
            continue
        if len(fuzzy) > 1:
            resolved.append(sorted(fuzzy)[0])
            continue
        missing.append(feature)
    if missing and not resolved:
        raise ValueError(
            "None of the requested CoDiet features could be resolved: "
            + ", ".join(missing)
        )
    if missing:
        print(f"codiet: ignoring unresolved requested features: {missing}")
    seen: set[str] = set()
    unique_resolved: list[str] = []
    for feature in resolved:
        if feature not in seen:
            unique_resolved.append(feature)
            seen.add(feature)
    return unique_resolved


def _build_graph_adjacency(selected_columns: list[str], graph: nx.DiGraph) -> np.ndarray:
    adjacency = np.zeros((len(selected_columns), len(selected_columns)), dtype=int)
    index = {column: idx for idx, column in enumerate(selected_columns)}
    present = [column for column in selected_columns if column in graph.nodes]
    if not present:
        return adjacency
    subgraph = graph.subgraph(present).copy()
    for src, dst in subgraph.edges():
        if src in index and dst in index:
            adjacency[index[src], index[dst]] = 1
    return adjacency


def _stage_graphml(raw_dir: Path, graph_path: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if graph_path.exists():
        target = raw_dir / graph_path.name
        if graph_path.resolve() != target.resolve():
            shutil.copy2(graph_path, target)


def build_codiet_bundle(
    raw_root: Path,
    processed_root: Path,
    cfg,
    output_name: str = "codiet",
) -> Path:
    """Materialize a CoDiet bundle under ``dataset/processed``.

    The function first tries the original source-repo style input:
    graphml + IPC/feather table + optional feature selection / sampling /
    scaling.  If those files are unavailable locally, it falls back to the
    local CoDiet bundle assembly built from the copied Recommender_Pavel data
    files.
    """

    raw_dir = raw_root / output_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg.get("data_path", CODIET_DATA_ROOT))
    data_filename = cfg.get("data_filename", "marks_data.feather")
    graph_filename = cfg.get(
        "knowledge_graph_filename",
        "knowledge_graph_intersection.graphml",
    )
    scale_data = cfg.get("scale_data", "quantile09")
    requested_features = cfg.get("features")
    n = cfg.get("n")
    target = cfg.get("target")

    graph_path = data_path / graph_filename
    table_path = data_path / data_filename

    if graph_path.exists() and table_path.exists():
        graph = nx.read_graphml(graph_path)
        df = _read_arrow_table(table_path)
        df = df.fillna(0)
        df = df.select_dtypes(include=["number"])
        if scale_data == "mean":
            denom = df.abs().mean().replace(0, 1.0)
            df = df / denom
        elif scale_data == "max":
            denom = df.abs().max().replace(0, 1.0)
            df = df / denom
        elif scale_data == "quantile09":
            denom = df.abs().quantile(0.95).replace(0, 1.0)
            df = df / denom
        else:
            raise ValueError(f"Unsupported codiet scale mode: {scale_data}")
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        if len(df) > 1:
            df = df.loc[:, [col for col in df.columns if df[col].nunique() > 1]]

        selected_columns = _resolve_requested_features(requested_features, df.columns)
        if target and target in df.columns and target not in selected_columns:
            selected_columns = selected_columns + [target]
        if "ID" in selected_columns and len(selected_columns) > 1:
            selected_columns = [column for column in selected_columns if column != "ID"]
        if n is not None:
            df = df.sample(n=int(n), random_state=int(cfg.get("seed", 42)), replace=False)
        if selected_columns:
            existing = [column for column in selected_columns if column in df.columns]
            if existing:
                df = df[existing]

        _stage_graphml(raw_dir, graph_path)
        df.to_csv(raw_dir / f"{output_name}.csv", index=False)
        adjacency = _build_graph_adjacency(list(df.columns), graph)
        np.save(raw_dir / "adj.npy", adjacency)
        np.save(raw_dir / "X.npy", df.to_numpy())
        np.save(raw_dir / "nodes.npy", np.asarray(df.columns))
        return _save_bundle(
            processed_root / output_name,
            x=df.to_numpy(),
            y=adjacency,
            nodes=df.columns,
            csv_name=f"{output_name}.csv",
        )

    _, _, prep_data = _load_codiet_local_bundle()
    graph = nx.read_graphml(CODIET_DATA_ROOT / "knowledge_graph_intersection.graphml")
    prep_data = prep_data.copy()
    prep_data = prep_data.loc[:, ~prep_data.columns.duplicated()]
    if "ID" in prep_data.columns:
        prep_data = prep_data.drop(columns=["ID"])

    selected_columns = _resolve_requested_features(requested_features, prep_data.columns)
    if target and target in prep_data.columns and target not in selected_columns:
        selected_columns = selected_columns + [target]
    if n is not None:
        prep_data = prep_data.sample(n=int(n), random_state=int(cfg.get("seed", 42)), replace=False)
    if selected_columns:
        existing = [column for column in selected_columns if column in prep_data.columns]
        if existing:
            prep_data = prep_data[existing]

    _stage_graphml(raw_dir, CODIET_DATA_ROOT / "knowledge_graph_intersection.graphml")
    prep_data.to_csv(raw_dir / f"{output_name}.csv", index=False)
    adjacency = _build_graph_adjacency(list(prep_data.columns), graph)
    np.save(raw_dir / "adj.npy", adjacency)
    np.save(raw_dir / "X.npy", prep_data.to_numpy())
    np.save(raw_dir / "nodes.npy", np.asarray(prep_data.columns))
    metadata = {
        "source": "dataset/raw/codiet",
        "target": target,
        "features": list(prep_data.columns),
    }
    with open(raw_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return _save_bundle(
        processed_root / output_name,
        x=prep_data.to_numpy(),
        y=adjacency,
        nodes=prep_data.columns,
        csv_name=f"{output_name}.csv",
    )
