#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plot_utils import configure_matplotlib, save_figure

from paths import DATASET_VISUALIZATIONS, FEATURES, PROCESSED

OUT = DATASET_VISUALIZATIONS
RAW_OUT = OUT / "merged"
SEL_OUT = OUT / "selected_features"
BOOK_NAME_MAP = {
    "万相": "Wanxiang",
    "元尊": "Yuan Zun",
    "大主宰": "The Great Ruler",
    "斗破": "Battle Through the Heavens",
    "武动": "Martial Universe",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    SEL_OUT.mkdir(parents=True, exist_ok=True)


def split_sents(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[。！？!?]+", str(text)) if x.strip()]


def translate_book_name(name: str) -> str:
    return BOOK_NAME_MAP.get(str(name), str(name))


def save_plot(fig, path: Path, *, left: float | None = None, bottom: float | None = None) -> None:
    save_figure(fig, path, dpi=160, left=left, bottom=bottom, right=0.98, top=0.93)


def visualize_merged() -> list[str]:
    df = pd.read_csv(PROCESSED / "merged.csv", encoding="utf-8-sig")
    df["paragragh"] = df["paragragh"].astype(str)
    df["char_len"] = df["paragragh"].str.len()
    df["n_sentences"] = df["paragragh"].apply(lambda x: max(len(split_sents(x)), 1))
    df["book"] = df["book"].astype(str)
    df["book_en"] = df["book"].map(translate_book_name)
    df["tag"] = df["tag"].astype(int)
    outputs: list[str] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    order = sorted(df["tag"].unique())
    sns.countplot(
        data=df,
        x="tag",
        hue="tag",
        order=order,
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Tag Distribution in merge.csv", pad=12)
    ax.set_xlabel("tag")
    ax.set_ylabel("Sample Count")
    save_plot(fig, RAW_OUT / "tag_distribution.png")
    outputs.append("merged/tag_distribution.png")

    book_tag = pd.crosstab(df["book_en"], df["tag"]).sort_index()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    book_tag.plot(kind="bar", stacked=True, colormap="tab20c", ax=ax)
    ax.set_title("Tag Distribution by Book in merge.csv", pad=12)
    ax.set_xlabel("book")
    ax.set_ylabel("Sample Count")
    ax.tick_params(axis="x", labelrotation=18)
    save_plot(fig, RAW_OUT / "book_tag_stacked_bar.png", bottom=0.16)
    outputs.append("merged/book_tag_stacked_bar.png")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    sns.boxplot(
        data=df,
        x="tag",
        y="char_len",
        hue="tag",
        palette="Set3",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Paragraph Length by Tag in merge.csv", pad=12)
    ax.set_xlabel("tag")
    ax.set_ylabel("Character Count")
    save_plot(fig, RAW_OUT / "char_len_by_tag_boxplot.png")
    outputs.append("merged/char_len_by_tag_boxplot.png")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    sns.histplot(
        data=df,
        x="char_len",
        hue="tag",
        bins=30,
        kde=True,
        stat="count",
        common_norm=False,
        ax=ax,
    )
    ax.set_title("Paragraph Length Distribution in merge.csv", pad=12)
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Frequency")
    save_plot(fig, RAW_OUT / "char_len_histogram.png")
    outputs.append("merged/char_len_histogram.png")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    sns.boxplot(
        data=df,
        x="tag",
        y="n_sentences",
        hue="tag",
        palette="Pastel1",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Sentence Count by Tag in merge.csv", pad=12)
    ax.set_xlabel("tag")
    ax.set_ylabel("Sentence Count")
    save_plot(fig, RAW_OUT / "sentence_count_by_tag_boxplot.png")
    outputs.append("merged/sentence_count_by_tag_boxplot.png")

    return outputs


def visualize_selected_features() -> list[str]:
    df = pd.read_csv(FEATURES / "NLP_Feature_train_selected.csv", encoding="utf-8-sig")
    df["label_tag"] = df["label_tag"].astype(int)
    feature_cols = [c for c in df.columns if c != "label_tag"]
    outputs: list[str] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.countplot(
        data=df,
        x="label_tag",
        hue="label_tag",
        order=sorted(df["label_tag"].unique()),
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Label Distribution in NLP_Feature_train_selected", pad=12)
    ax.set_xlabel("label_tag")
    ax.set_ylabel("Sample Count")
    save_plot(fig, SEL_OUT / "label_distribution.png")
    outputs.append("selected_features/label_distribution.png")

    variances = df[feature_cols].var().sort_values(ascending=False).head(20)
    variance_df = pd.DataFrame({"feature": variances.index, "variance": variances.values})
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    sns.barplot(
        data=variance_df,
        x="variance",
        y="feature",
        hue="feature",
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set_title("Top 20 Feature Variances After Selection", pad=12)
    ax.set_xlabel("variance")
    ax.set_ylabel("feature")
    ax.tick_params(axis="y", labelsize=9)
    save_plot(fig, SEL_OUT / "top20_feature_variances.png", left=0.30)
    outputs.append("selected_features/top20_feature_variances.png")

    corr_with_label = (
        df[feature_cols]
        .corrwith(df["label_tag"])
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(20)
    )
    corr_df = pd.DataFrame(
        {"feature": corr_with_label.index, "corr": corr_with_label.values}
    )
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    sns.barplot(data=corr_df, x="corr", y="feature", hue="corr", palette="vlag", legend=False, ax=ax)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title("Top 20 Feature Correlations with Label After Selection", pad=12)
    ax.set_xlabel("corr_with_label")
    ax.set_ylabel("feature")
    ax.tick_params(axis="y", labelsize=9)
    save_plot(fig, SEL_OUT / "top20_corr_with_label.png", left=0.30)
    outputs.append("selected_features/top20_corr_with_label.png")

    top_corr_features = corr_with_label.index[:15].tolist()
    heatmap_df = df[top_corr_features].corr()
    fig, ax = plt.subplots(figsize=(11.8, 10.2))
    sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        center=0,
        linewidths=0.2,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Highly Related Selected Features", pad=12)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    save_plot(fig, SEL_OUT / "feature_correlation_heatmap.png", left=0.24, bottom=0.18)
    outputs.append("selected_features/feature_correlation_heatmap.png")

    x_scaled = StandardScaler().fit_transform(df[feature_cols].values)
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(
        {
            "PC1": proj[:, 0],
            "PC2": proj[:, 1],
            "label_tag": df["label_tag"].astype(str),
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="label_tag",
        palette="Set1",
        alpha=0.8,
        ax=ax,
    )
    ax.set_title(
        f"PCA Projection of Selected Features (Explained Variance {pca.explained_variance_ratio_.sum():.2%})",
        pad=12,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save_plot(fig, SEL_OUT / "pca_scatter.png")
    outputs.append("selected_features/pca_scatter.png")

    return outputs


def main() -> None:
    chosen_font = configure_matplotlib()
    sns.set_theme(style="whitegrid", font=chosen_font)
    ensure_dirs()

    merged_outputs = visualize_merged()
    selected_outputs = visualize_selected_features()

    print("已生成以下可视化文件：")
    for rel_path in [*merged_outputs, *selected_outputs]:
        print(f"  - dataset_visualizations/{rel_path}")


if __name__ == "__main__":
    main()
