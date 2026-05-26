"""MIMAC 数据加载与列名自适应解析。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# 常见列名别名映射
ID_CANDIDATES = ["molecule_id", "mol_id", "id", "ID", "compound_id", "name"]
X_CANDIDATES = ["x", "umap_x", "X", "coord_x", "manifold_x", "dim1"]
Y_CANDIDATES = ["y", "umap_y", "Y", "coord_y", "manifold_y", "dim2"]
ACTIVITY_CANDIDATES = [
    "activity", "activity_score", "pActivity", "pIC50", "score", "label", "y_activity"
]
SOURCE_CANDIDATES = ["source", "node_1", "from", "u", "mol_i", "i"]
TARGET_CANDIDATES = ["target", "node_2", "to", "v", "mol_j", "j"]
WEIGHT_CANDIDATES = ["weight", "similarity", "sim", "edge_weight", "w"]


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"未找到列，候选: {candidates}，实际列: {list(df.columns)}")


def _auto_physicochemical_cols(df: pd.DataFrame, reserved: set[str]) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in reserved]


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_id = _pick_column(df, ID_CANDIDATES)
    col_x = _pick_column(df, X_CANDIDATES)
    col_y = _pick_column(df, Y_CANDIDATES)
    col_act = _pick_column(df, ACTIVITY_CANDIDATES)

    out = df.copy()
    out = out.rename(
        columns={
            col_id: "molecule_id",
            col_x: "x",
            col_y: "y",
            col_act: "activity",
        }
    )
    out["molecule_id"] = out["molecule_id"].astype(str)
    return out


def load_knn_graph(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_s = _pick_column(df, SOURCE_CANDIDATES)
    col_t = _pick_column(df, TARGET_CANDIDATES)
    out = df.rename(columns={col_s: "source", col_t: "target"})
    if any(c in df.columns for c in WEIGHT_CANDIDATES):
        col_w = _pick_column(df, WEIGHT_CANDIDATES)
        out = out.rename(columns={col_w: "weight"})
    else:
        out["weight"] = 1.0
    out["source"] = out["source"].astype(str)
    out["target"] = out["target"].astype(str)
    return out


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for row in edges.itertuples(index=False):
        g.add_edge(str(row.source), str(row.target), weight=float(row.weight))
    return g


def get_physicochemical_columns(df: pd.DataFrame) -> list[str]:
    reserved = {"molecule_id", "x", "y", "activity"}
    return _auto_physicochemical_cols(df, reserved)


def generate_synthetic_data(
    n_molecules: int = 220,
    k_neighbors: int = 8,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成与题意结构一致的模拟数据，便于无附件时演示全流程。"""
    rng = np.random.default_rng(seed)

    # 三个化学簇 + 一个高活性热点
    centers = np.array([[-2.5, -1.0], [2.0, 1.5], [0.0, -2.5], [2.8, -1.8]])
    cluster_sizes = [70, 60, 50, 40]
    cluster_sizes[-1] = n_molecules - sum(cluster_sizes[:-1])

    xs, ys, acts, ids = [], [], [], []
    for ci, (cx, cy) in enumerate(centers):
        n = cluster_sizes[ci]
        x = rng.normal(cx, 0.55, n)
        y = rng.normal(cy, 0.55, n)
        base = 5.5 + ci * 0.4
        if ci == 3:
            base = 8.2  # 热点簇更高活性
        activity = base + rng.normal(0, 0.35, n)
        # 在热点中植入活性悬崖：部分近邻活性突变
        if ci == 3:
            cliff_idx = rng.choice(n, size=max(6, n // 8), replace=False)
            activity[cliff_idx] -= rng.uniform(1.5, 3.0, size=len(cliff_idx))
        xs.extend(x)
        ys.extend(y)
        acts.extend(activity)
        ids.extend([f"M{i + 1:04d}" for i in range(len(ids), len(ids) + n)])

    manifest = pd.DataFrame(
        {
            "molecule_id": ids,
            "x": xs,
            "y": ys,
            "activity": acts,
            "mol_weight": rng.uniform(250, 520, n_molecules),
            "logP": rng.uniform(-1, 5, n_molecules),
            "tpsa": rng.uniform(20, 140, n_molecules),
            "hbd": rng.integers(0, 5, n_molecules),
            "hba": rng.integers(1, 10, n_molecules),
        }
    )

    coords = manifest[["x", "y"]].values
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    dists, idx = nn.kneighbors(coords)
    edge_rows = []
    for i in range(n_molecules):
        for j, d in zip(idx[i, 1:], dists[i, 1:], strict=False):
            sim = float(np.exp(-d ** 2 / 0.8))
            edge_rows.append(
                {
                    "source": manifest.loc[i, "molecule_id"],
                    "target": manifest.loc[j, "molecule_id"],
                    "weight": sim,
                }
            )
    edges = pd.DataFrame(edge_rows).drop_duplicates()

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out / "molecular_interaction_manifest.csv", index=False)
        edges.to_csv(out / "knn_graph_edges.csv", index=False)

    return manifest, edges


def load_or_synthesize(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, nx.Graph, bool]:
    data_dir = Path(data_dir)
    manifest_path = data_dir / "molecular_interaction_manifest.csv"
    edges_path = data_dir / "knn_graph_edges.csv"
    synthetic = False

    if manifest_path.exists() and edges_path.exists():
        manifest = load_manifest(manifest_path)
        edges = load_knn_graph(edges_path)
    else:
        synthetic = True
        manifest, edges = generate_synthetic_data(output_dir=data_dir)

    graph = build_graph(edges)
    return manifest, edges, graph, synthetic
