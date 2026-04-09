#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import argparse
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from plot_utils import configure_matplotlib, save_figure
from text_features_common import zh_word_unigram_bigram

from paths import ARTIFACTS, CV_RESULTS, EVALUATION, PROCESSED, TRAINING_FIGURES

ART = ARTIFACTS
ART.mkdir(exist_ok=True)
FIG = TRAINING_FIGURES
FIG.mkdir(parents=True, exist_ok=True)
CV_RESULTS.mkdir(parents=True, exist_ok=True)
EVALUATION.mkdir(parents=True, exist_ok=True)
configure_matplotlib()


def build_features_train_valid(x_train: list[str], x_valid: list[str]):
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=12000,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    word_vec = TfidfVectorizer(
        analyzer=zh_word_unigram_bigram,
        max_features=6000,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    xc_tr = char_vec.fit_transform(x_train)
    xw_tr = word_vec.fit_transform(x_train)
    xc_va = char_vec.transform(x_valid)
    xw_va = word_vec.transform(x_valid)
    x_tr = sparse.hstack([xc_tr, xw_tr], format="csr")
    x_va = sparse.hstack([xc_va, xw_va], format="csr")
    return char_vec, word_vec, x_tr, x_va


def pick_unknown_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    pred = proba.argmax(axis=1) + 1
    conf = proba.max(axis=1)
    ok = conf[pred == y_true]
    if len(ok) == 0:
        return 0.6
    # 保守一点：正确样本置信度的 10% 分位
    return float(np.quantile(ok, 0.10))


def fit_confidence_calibrator(y_true: np.ndarray, proba: np.ndarray) -> IsotonicRegression:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1) + 1
    is_correct = (pred == y_true).astype(float)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(conf, is_correct)
    return calibrator


def evaluate(name: str, model, x, y):
    p = model.predict(x)
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average="macro")
    bal_acc = balanced_accuracy_score(y, p)
    rec_tag2 = recall_score(y, p, labels=[2], average="macro", zero_division=0)
    return {
        "model": name,
        "acc": float(acc),
        "f1_macro": float(f1),
        "balanced_acc": float(bal_acc),
        "recall_tag2": float(rec_tag2),
    }


def train_and_tune_models(
    x_tr_sparse,
    x_tr_dense,
    y_tr: np.ndarray,
    random_state: int = 42,
    svm_iter: int = 10,
    rf_iter: int = 10,
    cv_splits: int = 4,
):
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    print("[2/6] 开始自动调参：SVM")

    svm_search = RandomizedSearchCV(
        estimator=SVC(
            kernel="linear",
            probability=True,
            random_state=random_state,
        ),
        param_distributions={"C": loguniform(1e-2, 1e2)},
        n_iter=svm_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    svm_search.fit(x_tr_sparse, y_tr, sample_weight=sample_weight)
    print(
        f"[2/6] SVM 调参完成，best_score={svm_search.best_score_:.4f}, best_params={svm_search.best_params_}"
    )

    print("[3/6] 开始自动调参：RandomForest")
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced_subsample",
            # 避免与 RandomizedSearchCV 的并行叠加导致过载
            n_jobs=1,
        ),
        param_distributions={
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        },
        n_iter=rf_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    rf_search.fit(x_tr_dense, y_tr, sample_weight=sample_weight)
    print(
        f"[3/6] RF 调参完成，best_score={rf_search.best_score_:.4f}, best_params={rf_search.best_params_}"
    )

    return svm_search, rf_search


def train_extra_models(
    x_tr_sparse,
    y_tr: np.ndarray,
    sample_weight: np.ndarray,
) -> dict:
    models: dict[str, object] = {}
    print("[3.5/6] 训练额外代表模型：MultinomialNB")
    nb = MultinomialNB(alpha=0.8)
    nb.fit(x_tr_sparse, y_tr, sample_weight=sample_weight)
    models["nb"] = nb
    return models


def pick_vote_weight(
    y_true: np.ndarray, proba_svm: np.ndarray, proba_rf: np.ndarray
) -> tuple[float, float, float]:
    best_score = -1.0
    best_w = (0.5, 0.5)
    for w_svm in np.linspace(0.2, 0.8, 13):
        w_rf = 1.0 - w_svm
        proba_mix = w_svm * proba_svm + w_rf * proba_rf
        pred = proba_mix.argmax(axis=1) + 1
        f1m = f1_score(y_true, pred, average="macro")
        rec2 = recall_score(y_true, pred, labels=[2], average="macro", zero_division=0)
        # 兼顾总体与小类召回，缓解不均衡下“看起来高分但漏检小类”的问题
        score = 0.8 * f1m + 0.2 * rec2
        if score > best_score:
            best_score = score
            best_w = (float(w_svm), float(w_rf))
    return best_w[0], best_w[1], float(best_score)


def save_training_figures(
    class_counts: pd.Series,
    eval_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls_report_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    class_counts.plot(kind="bar", color=["#4C78A8", "#F58518"], ax=ax)
    ax.set_title("Class Distribution (merged.csv)", pad=12)
    ax.set_xlabel("tag")
    ax.set_ylabel("count")
    save_figure(fig, FIG / "class_distribution.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.barplot(
        data=eval_df,
        x="model",
        y="f1_macro",
        hue="model",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Model F1-macro Comparison", pad=12)
    ax.tick_params(axis="x", labelrotation=12)
    save_figure(fig, FIG / "model_f1_comparison.png", dpi=160)

    cm = pd.crosstab(
        pd.Series(y_true, name="True"),
        pd.Series(y_pred, name="Pred"),
        dropna=False,
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix (weighted voting)", pad=12)
    save_figure(fig, FIG / "confusion_matrix_weighted_voting.png", dpi=160)

    metric_rows = [x for x in ["1", "2", "macro avg", "weighted avg"] if x in cls_report_df.index]
    metric_cols = [x for x in ["precision", "recall", "f1-score"] if x in cls_report_df.columns]
    if metric_rows and metric_cols:
        plot_df = cls_report_df.loc[metric_rows, metric_cols]
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
        ax.set_title("Classification Metrics Overview", pad=12)
        save_figure(fig, FIG / "classification_metrics_overview.png", dpi=160)


def dataframe_to_markdown_fallback(df: pd.DataFrame, float_digits: int = 4) -> str:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(
            lambda x: f"{x:.{float_digits}f}" if isinstance(x, (float, np.floating)) else str(x)
        )
    headers = ["index"] + [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in out.iterrows():
        vals = [str(idx)] + [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练文本分类模型（自动调参）")
    parser.add_argument("--quick", action="store_true", help="快速模式（更少搜索轮次）")
    parser.add_argument("--svm-iter", type=int, default=None, help="SVM 随机搜索轮次")
    parser.add_argument("--rf-iter", type=int, default=None, help="RF 随机搜索轮次")
    parser.add_argument("--cv", type=int, default=None, help="交叉验证折数")
    args = parser.parse_args()

    if args.quick:
        svm_iter = 6
        rf_iter = 6
        cv_splits = 3
    else:
        svm_iter = 10
        rf_iter = 10
        cv_splits = 4
    if args.svm_iter is not None:
        svm_iter = int(args.svm_iter)
    if args.rf_iter is not None:
        rf_iter = int(args.rf_iter)
    if args.cv is not None:
        cv_splits = int(args.cv)

    t0 = time.time()
    print(
        f"[0/6] 调参配置: svm_iter={svm_iter}, rf_iter={rf_iter}, cv={cv_splits}"
    )
    print("[1/6] 读取 merged.csv 并检查样本分布")
    df = pd.read_csv(PROCESSED / "merged.csv", encoding="utf-8-sig")
    texts = df["paragragh"].astype(str).tolist()
    y = df["tag"].astype(int).values
    class_counts = pd.Series(y).value_counts().sort_index()
    print("[1/6] 样本分布：")
    for k, v in class_counts.items():
        print(f"  - tag{k}: {v}")
    max_cnt = int(class_counts.max())
    min_cnt = int(class_counts.min())
    ratio = max_cnt / max(min_cnt, 1)
    print(f"[1/6] 最大/最小类别样本比: {ratio:.2f}")

    print("[4/6] 划分训练/验证并构建特征")
    x_tr_txt, x_va_txt, y_tr, y_va = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )
    char_vec, word_vec, x_tr, x_va = build_features_train_valid(x_tr_txt, x_va_txt)
    x_tr_dense = x_tr.toarray()
    x_va_dense = x_va.toarray()

    svm_search, rf_search = train_and_tune_models(
        x_tr,
        x_tr_dense,
        y_tr,
        random_state=42,
        svm_iter=svm_iter,
        rf_iter=rf_iter,
        cv_splits=cv_splits,
    )
    svm = svm_search.best_estimator_
    rf = rf_search.best_estimator_
    sample_weight_tr = compute_sample_weight(class_weight="balanced", y=y_tr)
    extra_models = train_extra_models(
        x_tr,
        y_tr,
        sample_weight=sample_weight_tr,
    )

    svm_proba_va = svm.predict_proba(x_va)
    rf_proba_va = rf.predict_proba(x_va_dense)
    print("[5/6] 计算验证集融合权重与评估")
    vote_w_svm, vote_w_rf, vote_pick_score = pick_vote_weight(y_va, svm_proba_va, rf_proba_va)
    voting_proba_va = vote_w_svm * svm_proba_va + vote_w_rf * rf_proba_va
    voting_pred_va = voting_proba_va.argmax(axis=1) + 1
    print(
        f"[5/6] 融合权重：w_svm={vote_w_svm:.2f}, w_rf={vote_w_rf:.2f}, pick_score={vote_pick_score:.4f}"
    )

    eval_rows = [
        evaluate("svm", svm, x_va, y_va),
        evaluate("rf", rf, x_va_dense, y_va),
        {
            "model": "weighted_voting",
            "acc": float(accuracy_score(y_va, voting_pred_va)),
            "f1_macro": float(f1_score(y_va, voting_pred_va, average="macro")),
            "balanced_acc": float(balanced_accuracy_score(y_va, voting_pred_va)),
            "recall_tag2": float(
                recall_score(y_va, voting_pred_va, labels=[2], average="macro", zero_division=0)
            ),
        },
    ]
    if "nb" in extra_models:
        eval_rows.append(evaluate("nb", extra_models["nb"], x_va, y_va))
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(EVALUATION / "model_eval.csv", index=False, encoding="utf-8-sig")
    print("[5/6] 各模型验证表现：")
    print(eval_df.to_string(index=False))

    unknown_threshold = pick_unknown_threshold(y_va, voting_proba_va)
    score_calibrator = fit_confidence_calibrator(y_va, voting_proba_va)
    calibrated_conf_va = score_calibrator.predict(voting_proba_va.max(axis=1))

    report = classification_report(y_va, voting_pred_va, digits=4)
    with (EVALUATION / "model_eval_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)
    print("[5/6] Classification report:")
    print(report)

    cls_report_dict = classification_report(y_va, voting_pred_va, output_dict=True, digits=4)
    cls_report_df = pd.DataFrame(cls_report_dict).T
    cls_report_df.to_csv(EVALUATION / "classification_report_table.csv", encoding="utf-8-sig")
    with (EVALUATION / "classification_report_table.md").open("w", encoding="utf-8") as f:
        f.write("# Classification Report Table\n\n")
        try:
            f.write(cls_report_df.to_markdown())
        except ImportError:
            f.write(dataframe_to_markdown_fallback(cls_report_df, float_digits=4))

    joblib.dump(char_vec, ART / "char_vec.joblib")
    joblib.dump(word_vec, ART / "word_vec.joblib")
    joblib.dump(svm, ART / "svm_model.joblib")
    joblib.dump(rf, ART / "rf_model.joblib")
    if "nb" in extra_models:
        joblib.dump(extra_models["nb"], ART / "nb_model.joblib")
    # 保存推理所需的融合配置（demo 将按该配置进行软投票）
    fusion_config = {"w_svm": vote_w_svm, "w_rf": vote_w_rf}
    joblib.dump(fusion_config, ART / "voting_model.joblib")
    joblib.dump(score_calibrator, ART / "score_calibrator.joblib")

    meta = {
        "labels": {"1": "tag1", "2": "tag2", "0": "none"},
        "unknown_threshold": unknown_threshold,
        "score_mode": "calibrated_vote_confidence",
        "n_train": int(len(y_tr)),
        "n_valid": int(len(y_va)),
        "class_distribution_full": {str(int(k)): int(v) for k, v in class_counts.items()},
        "best_params": {
            "svm": svm_search.best_params_,
            "rf": rf_search.best_params_,
        },
        "vote_weights": fusion_config,
        "enabled_models": ["svm", "rf", "weighted_voting"]
        + (["nb"] if "nb" in extra_models else []),
        "score_calibration_preview": {
            "raw_conf_mean": float(voting_proba_va.max(axis=1).mean()),
            "calibrated_conf_mean": float(np.mean(calibrated_conf_va)),
        },
    }
    with (ART / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_training_figures(class_counts, eval_df, y_va, voting_pred_va, cls_report_df)
    eval_df.sort_values("f1_macro", ascending=False).to_csv(
        EVALUATION / "model_eval_sorted.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(svm_search.cv_results_).to_csv(
        CV_RESULTS / "svm_cv_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rf_search.cv_results_).to_csv(
        CV_RESULTS / "rf_cv_results.csv", index=False, encoding="utf-8-sig"
    )

    elapsed = time.time() - t0
    print("[6/6] 训练完成")
    print(f"  - none 阈值: {unknown_threshold:.4f}")
    print(f"  - 产物目录: {ART}")
    print(f"  - 图像目录: {FIG}")
    print(f"  - 用时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
