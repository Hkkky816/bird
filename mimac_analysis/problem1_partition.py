"""问题1：流形空间分区 — KMeans++ 增强版 vs KDE/GMM 对比。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, zscore
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from data_loader import get_physicochemical_columns


@dataclass
class PartitionResult:
    method: str
    labels: np.ndarray
    cluster_hotspot: dict[int, bool]
    molecule_region: np.ndarray  # 'hotspot' | 'normal'
    representatives: dict[str, list[str]]
    cluster_stats: pd.DataFrame
    metrics: dict[str, float]


def _standardize_features(manifest: pd.DataFrame, activity_weight: float = 0.8) -> np.ndarray:
    coords = manifest[["x", "y"]].values.astype(float)
    activity = manifest["activity"].values.astype(float).reshape(-1, 1)
    z_coord = zscore(coords, axis=0, ddof=0)
    z_act = zscore(activity, axis=0, ddof=0)
    return np.hstack([z_coord, activity_weight * z_act])


def choose_k_by_silhouette(features: np.ndarray, k_min: int = 3, k_max: int = 8) -> int:
    best_k, best_score = k_min, -1.0
    n = len(features)
    upper = min(k_max, n - 1)
    for k in range(k_min, upper + 1):
        labels = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit_predict(features)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(features, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _activity_hotspot_ratio(activity: np.ndarray, threshold: float) -> float:
    return float(np.mean(activity >= threshold))


def classify_clusters(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    graph: nx.Graph,
    density_scores: np.ndarray | None = None,
) -> tuple[dict[int, bool], np.ndarray, pd.DataFrame]:
    """根据活性、高活性占比、图连通性、密度判定热点簇。"""
    global_mean = manifest["activity"].mean()
    global_std = manifest["activity"].std(ddof=0)
    high_threshold = float(np.quantile(manifest["activity"], 0.75))

    cluster_hotspot: dict[int, bool] = {}
    rows = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        sub = manifest.loc[mask]
        act_mean = sub["activity"].mean()
        act_std = sub["activity"].std(ddof=0)
        hpi = _activity_hotspot_ratio(sub["activity"].values, high_threshold)

        degrees = [graph.degree[n] if n in graph else 0 for n in sub["molecule_id"]]
        avg_degree = float(np.mean(degrees)) if degrees else 0.0

        local_density = float(np.mean(density_scores[mask])) if density_scores is not None else np.nan

        votes = 0
        if act_mean > global_mean + 0.5 * global_std:
            votes += 1
        if hpi > 0.4:
            votes += 1
        if not np.isnan(local_density) and local_density > np.nanmedian(density_scores):
            votes += 1

        is_hotspot = votes >= 2
        cluster_hotspot[int(cluster_id)] = is_hotspot
        rows.append(
            {
                "cluster": int(cluster_id),
                "n_molecules": int(mask.sum()),
                "activity_mean": act_mean,
                "activity_std": act_std,
                "high_activity_ratio": hpi,
                "avg_graph_degree": avg_degree,
                "local_density": local_density,
                "is_hotspot": is_hotspot,
            }
        )

    stats = pd.DataFrame(rows)
    molecule_region = np.array(
        ["hotspot" if cluster_hotspot[int(c)] else "normal" for c in labels]
    )
    return cluster_hotspot, molecule_region, stats


def activity_weighted_kde(manifest: pd.DataFrame, grid_size: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """活性加权核密度估计，返回每个分子的局部密度得分。"""
    x = manifest["x"].values
    y = manifest["y"].values
    activity = manifest["activity"].values

    act_min, act_max = activity.min(), activity.max()
    weights = (activity - act_min) / (act_max - act_min + 1e-8) + 0.2

    sample = np.vstack([x, y])
    kde = gaussian_kde(sample, weights=weights)
    density_points = kde(sample)
    return density_points, kde


def partition_kmeans_plusplus(
    manifest: pd.DataFrame,
    graph: nx.Graph,
    n_clusters: int | None = None,
    activity_weight: float = 0.8,
) -> PartitionResult:
    features = _standardize_features(manifest, activity_weight=activity_weight)
    k = n_clusters or choose_k_by_silhouette(features)
    labels = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42).fit_predict(features)

    density_scores, _ = activity_weighted_kde(manifest)
    cluster_hotspot, molecule_region, cluster_stats = classify_clusters(
        manifest, labels, graph, density_scores
    )
    reps = pick_representatives(manifest, labels, cluster_hotspot, graph)
    metrics = {
        "silhouette": float(silhouette_score(features, labels)),
        "n_clusters": float(k),
        "hotspot_clusters": float(sum(cluster_hotspot.values())),
    }
    return PartitionResult(
        method="KMeans++ (x,y,λA)",
        labels=labels,
        cluster_hotspot=cluster_hotspot,
        molecule_region=molecule_region,
        representatives=reps,
        cluster_stats=cluster_stats,
        metrics=metrics,
    )


def partition_gmm(
    manifest: pd.DataFrame,
    graph: nx.Graph,
    n_components: int | None = None,
) -> PartitionResult:
    features = _standardize_features(manifest, activity_weight=1.0)
    k = n_components or choose_k_by_silhouette(features)
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)
    labels = gmm.fit_predict(features)

    density_scores, _ = activity_weighted_kde(manifest)
    cluster_hotspot, molecule_region, cluster_stats = classify_clusters(
        manifest, labels, graph, density_scores
    )
    reps = pick_representatives(manifest, labels, cluster_hotspot, graph)
    metrics = {
        "silhouette": float(silhouette_score(features, labels)),
        "n_clusters": float(k),
        "hotspot_clusters": float(sum(cluster_hotspot.values())),
        "bic": float(gmm.bic(features)),
    }
    return PartitionResult(
        method="GMM (x,y,A)",
        labels=labels,
        cluster_hotspot=cluster_hotspot,
        molecule_region=molecule_region,
        representatives=reps,
        cluster_stats=cluster_stats,
        metrics=metrics,
    )


def partition_kde_dbscan_style(
    manifest: pd.DataFrame,
    graph: nx.Graph,
    n_clusters: int | None = None,
) -> PartitionResult:
    """KDE 密度分层 + KMeans++ 空间细化（对照方案）。"""
    density_scores, _ = activity_weighted_kde(manifest)
    density_z = zscore(density_scores, ddof=0).reshape(-1, 1)
    spatial = zscore(manifest[["x", "y"]].values, axis=0, ddof=0)
    features = np.hstack([spatial, 0.6 * density_z])

    k = n_clusters or choose_k_by_silhouette(features)
    labels = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42).fit_predict(features)

    cluster_hotspot, molecule_region, cluster_stats = classify_clusters(
        manifest, labels, graph, density_scores
    )
    reps = pick_representatives(manifest, labels, cluster_hotspot, graph)
    metrics = {
        "silhouette": float(silhouette_score(features, labels)),
        "n_clusters": float(k),
        "hotspot_clusters": float(sum(cluster_hotspot.values())),
    }
    return PartitionResult(
        method="KDE + KMeans++",
        labels=labels,
        cluster_hotspot=cluster_hotspot,
        molecule_region=molecule_region,
        representatives=reps,
        cluster_stats=cluster_stats,
        metrics=metrics,
    )


def pick_representatives(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    cluster_hotspot: dict[int, bool],
    graph: nx.Graph,
) -> dict[str, list[str]]:
    """每个簇选代表分子：热点选高活性+近中心；普通选近均值；桥接选高介数。"""
    reps: dict[str, list[str]] = {"hotspot": [], "normal": [], "bridge": []}
    coords = manifest[["x", "y"]].values

    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        sub = manifest.loc[mask].copy()
        sub_coords = coords[mask]
        center = sub_coords.mean(axis=0)
        dist_to_center = np.linalg.norm(sub_coords - center, axis=1)

        if cluster_hotspot[int(cluster_id)]:
            rank = sub["activity"].values - 0.3 * dist_to_center
            best_idx = int(np.argmax(rank))
            reps["hotspot"].append(sub.iloc[best_idx]["molecule_id"])
        else:
            act_gap = np.abs(sub["activity"].values - sub["activity"].mean())
            rank = -act_gap - 0.2 * dist_to_center
            best_idx = int(np.argmax(rank))
            reps["normal"].append(sub.iloc[best_idx]["molecule_id"])

    # 桥接分子：度数高且活性居中
    degrees = manifest["molecule_id"].map(lambda m: graph.degree[m] if m in graph else 0)
    bridge_score = degrees - 0.5 * np.abs(zscore(manifest["activity"].values, ddof=0))
    top_bridge = manifest.iloc[np.argsort(-bridge_score.values)[:3]]["molecule_id"].tolist()
    reps["bridge"] = top_bridge
    return reps


def compare_regions(
    manifest: pd.DataFrame,
    molecule_region: np.ndarray,
    graph: nx.Graph,
) -> pd.DataFrame:
    """比较热点区 vs 普通区在活性、图结构、理化指标上的差异。"""
    df = manifest.copy()
    df["region"] = molecule_region
    physchem = get_physicochemical_columns(df)

    rows = []
    for region in ["hotspot", "normal"]:
        sub = df[df["region"] == region]
        if sub.empty:
            continue
        row = {
            "region": region,
            "count": len(sub),
            "activity_mean": sub["activity"].mean(),
            "activity_std": sub["activity"].std(ddof=0),
        }
        degrees = [graph.degree[m] if m in graph else 0 for m in sub["molecule_id"]]
        row["avg_degree"] = float(np.mean(degrees))
        for col in physchem:
            row[f"{col}_mean"] = sub[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compare_methods(results: list[PartitionResult]) -> pd.DataFrame:
    """横向对比三种分区方法。"""
    rows = []
    for r in results:
        row = {"method": r.method}
        row.update(r.metrics)
        row["hotspot_molecules"] = int(np.sum(r.molecule_region == "hotspot"))
        row["normal_molecules"] = int(np.sum(r.molecule_region == "normal"))
        rows.append(row)

    comp = pd.DataFrame(rows)

    # 方法间热点标签一致性（ARI）
    if len(results) >= 2:
        hotspot_labels = [
            (res.molecule_region == "hotspot").astype(int) for res in results
        ]
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                ari = adjusted_rand_score(hotspot_labels[i], hotspot_labels[j])
                comp.loc[len(comp)] = {
                    "method": f"ARI({results[i].method} vs {results[j].method})",
                    "silhouette": ari,
                }
    return comp


def plot_partitions(
    manifest: pd.DataFrame,
    results: list[PartitionResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 4.5))
    if n + 1 == 1:
        axes = [axes]

    density, _ = activity_weighted_kde(manifest)
    ax0 = axes[0]
    sc = ax0.scatter(
        manifest["x"],
        manifest["y"],
        c=density,
        cmap="YlOrRd",
        s=28,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.2,
    )
    ax0.set_title("Activity-weighted KDE")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    fig.colorbar(sc, ax=ax0, fraction=0.046)

    for ax, result in zip(axes[1:], results, strict=True):
        region_colors = np.where(result.molecule_region == "hotspot", 1, 0)
        ax.scatter(
            manifest["x"],
            manifest["y"],
            c=result.labels,
            cmap="tab10",
            s=22,
            alpha=0.35,
            edgecolors="none",
        )
        hotspot_mask = result.molecule_region == "hotspot"
        ax.scatter(
            manifest.loc[hotspot_mask, "x"],
            manifest.loc[hotspot_mask, "y"],
            c="red",
            s=40,
            marker="*",
            label="hotspot",
        )
        ax.set_title(result.method)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "problem1_partition_comparison.png", dpi=160)
    plt.close(fig)


def run_problem1(
    manifest: pd.DataFrame,
    graph: nx.Graph,
    output_dir: str | Path,
    n_clusters: int | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kmeans_result = partition_kmeans_plusplus(manifest, graph, n_clusters=n_clusters)
    gmm_result = partition_gmm(manifest, graph, n_components=n_clusters)
    kde_result = partition_kde_dbscan_style(manifest, graph, n_clusters=n_clusters)

    results = [kmeans_result, gmm_result, kde_result]
    method_comparison = compare_methods(results)

    # 以 KMeans++ 增强版作为问题1主结果
    primary = kmeans_result
    region_comparison = compare_regions(manifest, primary.molecule_region, graph)

    manifest_out = manifest.copy()
    manifest_out["cluster"] = primary.labels
    manifest_out["region"] = primary.molecule_region

    manifest_out.to_csv(output_dir / "problem1_molecule_labels.csv", index=False)
    primary.cluster_stats.to_csv(output_dir / "problem1_cluster_stats.csv", index=False)
    region_comparison.to_csv(output_dir / "problem1_region_comparison.csv", index=False)
    method_comparison.to_csv(output_dir / "problem1_method_comparison.csv", index=False)

    plot_partitions(manifest, results, output_dir)

    summary = {
        "primary_method": primary.method,
        "n_clusters": int(primary.metrics["n_clusters"]),
        "hotspot_clusters": int(primary.metrics["hotspot_clusters"]),
        "representatives": primary.representatives,
        "method_comparison": method_comparison,
        "region_comparison": region_comparison,
    }
    return summary
