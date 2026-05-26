"""运行问题1：KMeans++ 增强版 vs KDE/GMM 对比。"""

from __future__ import annotations

from pathlib import Path

from data_loader import load_or_synthesize
from problem1_partition import run_problem1


def main() -> None:
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    output_dir = base / "output"

    manifest, edges, graph, synthetic = load_or_synthesize(data_dir)
    if synthetic:
        print(f"[提示] 未找到附件，已生成模拟数据: {data_dir}")

    summary = run_problem1(manifest, graph, output_dir)

    print("\n========== 问题1：流形空间分区 ==========")
    print(f"主方案: {summary['primary_method']}")
    print(f"簇数 K = {summary['n_clusters']}")
    print(f"热点簇数量 = {summary['hotspot_clusters']}")

    print("\n--- 代表分子 ---")
    for kind, ids in summary["representatives"].items():
        print(f"  {kind}: {', '.join(ids)}")

    print("\n--- 热点 vs 普通区 对比 ---")
    print(summary["region_comparison"].to_string(index=False))

    print("\n--- 方法对比 (KMeans++ / GMM / KDE+KMeans++) ---")
    print(summary["method_comparison"].to_string(index=False))

    print(f"\n结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
