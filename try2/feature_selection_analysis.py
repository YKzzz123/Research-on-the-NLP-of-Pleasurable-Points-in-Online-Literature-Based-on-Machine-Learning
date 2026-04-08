# -*- coding: utf-8 -*-
"""
基于 NLP_Feature_train.csv 做特征与标签相关度 / 重要性分析，并导出可选特征子集。

分析内容：
  - 与 label_tag 的 Pearson 相关系数（二分类标签下等价于点双列相关的排序）
  - 互信息 mutual_info_classif（捕捉非线性）
  - ANOVA F 值与 p 值（f_classif）
  - 特征间高相关对（|r|>阈值，提示冗余）

输出（均在 try2/）：
  - feature_scores.csv
  - feature_redundancy_pairs.csv
  - selected_feature_list.json
  - NLP_Feature_train_selected.csv（综合得分 Top-K 特征 + label_tag）
  - corr_label_top_features.png（与标签相关性最高的若干特征柱状图）
  - corr_feature_heatmap_top.png（特征间相关热力图，Top 子集）

运行：
  conda activate comp5572
  python try2/feature_selection_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from plot_utils import configure_matplotlib, save_figure

TRY2 = Path(__file__).resolve().parent
BOOK_FEATURE_MAP = {
    "book_万相": "book_Wanxiang",
    "book_元尊": "book_YuanZun",
    "book_大主宰": "book_TheGreatRuler",
    "book_斗破": "book_BattleThroughTheHeavens",
    "book_武动": "book_MartialUniverse",
}

# 导出「精简训练集」时保留的特征个数（可调）
TOP_K = 80
# 标记特征间强相关（冗余）
REDUNDANCY_THRESHOLD = 0.88


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    chosen_font = configure_matplotlib()
    sns.set_theme(style="whitegrid", font=chosen_font)
    from sklearn.feature_selection import f_classif, mutual_info_classif

    train_path = TRY2 / "NLP_Feature_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"请先运行 build_nlp_features.py 生成 {train_path}")

    df = pd.read_csv(train_path, encoding="utf-8-sig")
    y = df["label_tag"].astype(int).values
    X_df = df.drop(columns=["label_tag"])
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float64)

    n = len(y)
    # 与标签的 Pearson 相关（逐列）
    corr_label = np.array(
        [
            np.corrcoef(X[:, j], y)[0, 1]
            if np.std(X[:, j]) > 1e-12
            else 0.0
            for j in range(X.shape[1])
        ]
    )
    corr_abs = np.abs(corr_label)

    mi = mutual_info_classif(X, y, random_state=42)
    f_vals, p_vals = f_classif(X, y)

    scores_df = pd.DataFrame(
        {
            "feature": feature_names,
            "corr_with_label": corr_label,
            "abs_corr_with_label": corr_abs,
            "mutual_info": mi,
            "f_classif": f_vals,
            "p_value_f_classif": p_vals,
        }
    )
    scores_df = scores_df.sort_values("mutual_info", ascending=False).reset_index(
        drop=True
    )

    # 综合排序：互信息与 |相关| 的秩平均（0~1 归一化后）
    def _minmax(a: np.ndarray) -> np.ndarray:
        lo, hi = np.min(a), np.max(a)
        if hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    def _display_feature_name(name: str) -> str:
        return BOOK_FEATURE_MAP.get(str(name), str(name))

    rank_mi = _minmax(mi)
    rank_corr = _minmax(corr_abs)
    rank_f = _minmax(f_vals)
    combined = (rank_mi + rank_corr + rank_f) / 3.0
    scores_df["score_combined"] = combined
    scores_df = scores_df.sort_values("score_combined", ascending=False).reset_index(
        drop=True
    )

    scores_path = TRY2 / "feature_scores.csv"
    scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")

    # 特征间 Pearson 相关矩阵（仅用于找冗余对）
    R = np.corrcoef(X.T)
    np.fill_diagonal(R, 0.0)
    pairs = []
    p = R.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            r = R[i, j]
            if np.isnan(r):
                continue
            if abs(r) >= REDUNDANCY_THRESHOLD:
                pairs.append(
                    {
                        "feature_i": feature_names[i],
                        "feature_j": feature_names[j],
                        "corr_ij": float(r),
                    }
                )
    red_df = pd.DataFrame(pairs)
    if len(red_df):
        red_df = red_df.assign(_abs=np.abs(red_df["corr_ij"])).sort_values(
            "_abs", ascending=False
        ).drop(columns=["_abs"])
    red_path = TRY2 / "feature_redundancy_pairs.csv"
    red_df.to_csv(red_path, index=False, encoding="utf-8-sig")

    # 取 Top-K（按 combined）
    top_features = scores_df["feature"].head(TOP_K).tolist()
    selected_json = {
        "top_k": TOP_K,
        "selection_rule": "按 score_combined = mean(秩归一化(mutual_info), abs_corr, f_classif)) 排序",
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "features": top_features,
    }
    with (TRY2 / "selected_feature_list.json").open("w", encoding="utf-8") as f:
        json.dump(selected_json, f, ensure_ascii=False, indent=2)

    sub = df[top_features + ["label_tag"]]
    sub_path = TRY2 / "NLP_Feature_train_selected.csv"
    sub.to_csv(sub_path, index=False, encoding="utf-8-sig")

    # 图1：与标签 |相关| 最高的 20 个特征
    top20 = scores_df.nlargest(20, "abs_corr_with_label")
    top20_plot = top20.assign(feature_display=top20["feature"].map(_display_feature_name))
    fig, ax = plt.subplots(figsize=(12.5, 7.8))
    sns.barplot(
        data=top20_plot,
        y="feature_display",
        x="corr_with_label",
        hue="corr_with_label",
        palette="vlag",
        legend=False,
        ax=ax,
    )
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title("Pearson Correlation with label_tag (Top 20 by Absolute Value)", pad=14)
    ax.set_xlabel("corr_with_label")
    ax.set_ylabel("feature")
    ax.tick_params(axis="y", labelsize=9)
    save_figure(
        fig,
        TRY2 / "corr_label_top_features.png",
        dpi=150,
        left=0.30,
        right=0.98,
        top=0.92,
        bottom=0.10,
    )

    # 图2：与标签相关最高的前 25 个特征之间的相关热力图
    n_hm = min(25, len(feature_names))
    top_for_hm = scores_df.nlargest(n_hm, "abs_corr_with_label")["feature"].tolist()
    top_for_hm_display = [_display_feature_name(name) for name in top_for_hm]
    idx = [feature_names.index(c) for c in top_for_hm]
    sub_R = np.corrcoef(X[:, idx].T)
    fig, ax = plt.subplots(figsize=(15, 12.5))
    sns.heatmap(
        sub_R,
        xticklabels=top_for_hm_display,
        yticklabels=top_for_hm_display,
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.2,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix of Top Label-Related Features", pad=14)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    save_figure(
        fig,
        TRY2 / "corr_feature_heatmap_top.png",
        dpi=150,
        left=0.24,
        right=0.98,
        top=0.93,
        bottom=0.20,
    )

    print(
        f"已写入：\n"
        f"  {scores_path.name}（全特征得分，共 {len(scores_df)} 列特征）\n"
        f"  {red_path.name}（强相关特征对 {len(red_df)} 条）\n"
        f"  selected_feature_list.json\n"
        f"  {sub_path.name}（{TOP_K} 维 + label_tag）\n"
        f"  corr_label_top_features.png\n"
        f"  corr_feature_heatmap_top.png"
    )


if __name__ == "__main__":
    main()
