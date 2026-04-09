# -*- coding: utf-8 -*-
"""
读取 merged.csv，构建面向「爽感」文本的 NLP 特征表（含 SnowNLP 情感）。

在 comp5572 虚拟环境中运行：
  conda activate comp5572
  python try2/build_nlp_features.py

输出：
  - NLP_Feature_Matrix.csv：含 idx / book / chapter，便于查看
  - NLP_Feature_train.csv：仅数值列 + label_tag，可直接喂给 RandomForest / SVM（见下）
  - NLP_Feature_train_scaled.csv：对连续特征做 StandardScaler（树模型不必用；线性 SVM 建议用）
  - NLP_Feature_vocab.json：TF-IDF 列名与 ngram 对应
  - NLP_Feature_train_columns.json：训练用列名顺序说明

训练示例（随机森林）：
  import pandas as pd
  df = pd.read_csv("try2/data/features/NLP_Feature_train.csv", encoding="utf-8-sig")
  y = df["label_tag"].astype(int)
  X = df.drop(columns=["label_tag"])

训练示例（线性 SVM，建议用缩放矩阵）：
  df = pd.read_csv("try2/data/features/NLP_Feature_train_scaled.csv", encoding="utf-8-sig")
  y = df["label_tag"].astype(int)
  X = df.drop(columns=["label_tag"])
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import jieba
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from snownlp import SnowNLP

from paths import FEATURES, PROCESSED

LEX_UGC = frozenset(
    "震惊 骇然 不可思议 目瞪口呆 哗然 倒吸 碾压 摧枯拉朽 狼狈 惊骇 恐怖 可怕 强悍 变态 妖孽 怪物".split()
)
LEX_POSS = frozenset(
    "得到 获得 收入 囊中 到手 机缘 宝物 传承 精血 吞纳 炼化 突破 暴涨 金手指".split()
)


def split_sents(t: str) -> list[str]:
    return [x.strip() for x in re.split(r"[。！？]+", t) if x.strip()]


def count_hits(t: str, vocab: frozenset) -> int:
    w = set(jieba.lcut(t))
    return sum(1 for x in vocab if x in w)


def zh_word_unigram_bigram(doc: str) -> list[str]:
    toks = jieba.lcut(doc)
    out = list(toks)
    for i in range(len(toks) - 1):
        out.append(f"{toks[i]}::{toks[i + 1]}")
    return out


def snownlp_sentiment(text: str) -> tuple[float, float]:
    """
    SnowNLP 情感：整段分数 [0,1]（越高越偏积极）；
    句级标准差：按句抽样至多 8 句算 std，刻画段落内情感起伏（控制耗时）。
    """
    if not text or not str(text).strip():
        return 0.5, 0.0
    t = str(text).strip()
    try:
        doc_score = float(SnowNLP(t).sentiments)
    except Exception:
        doc_score = 0.5
    sents = [
        x.strip()
        for x in re.split(r"[。！？\n]+", t)
        if x.strip() and len(x.strip()) >= 4
    ]
    if len(sents) < 2:
        return doc_score, 0.0
    step = max(1, len(sents) // 8)
    sample = sents[::step][:8]
    vals: list[float] = []
    for s in sample:
        try:
            vals.append(float(SnowNLP(s).sentiments))
        except Exception:
            continue
    if len(vals) < 2:
        return doc_score, 0.0
    return doc_score, float(np.std(vals, dtype=np.float64))


def _sanitize(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def build_features(
    texts: list[str], tags: list[str]
) -> tuple[dict[str, np.ndarray], dict, list[str]]:
    n = len(texts)
    char_lens = np.array([len(t) for t in texts], dtype=np.float64)
    word_counts = np.array([len(jieba.lcut(t)) for t in texts], dtype=np.float64)

    n_sents: list[float] = []
    mean_sent_len: list[float] = []
    excl: list[float] = []
    quest: list[float] = []
    ugc_hits: list[float] = []
    poss_hits: list[float] = []
    sent_doc: list[float] = []
    sent_std: list[float] = []

    for t in texts:
        sents = split_sents(t)
        n_sents.append(float(len(sents) if sents else 1))
        mean_sent_len.append(
            float(np.mean([len(s) for s in sents])) if sents else 0.0
        )
        excl.append(float(t.count("!") + t.count("！")))
        quest.append(float(t.count("?") + t.count("？")))
        ugc_hits.append(float(count_hits(t, LEX_UGC)))
        poss_hits.append(float(count_hits(t, LEX_POSS)))
        sd, ss = snownlp_sentiment(t)
        sent_doc.append(sd)
        sent_std.append(ss)

    n_sents_arr = _sanitize(np.array(n_sents))
    mean_sent_len_arr = _sanitize(np.array(mean_sent_len))
    excl_arr = _sanitize(np.array(excl))
    quest_arr = _sanitize(np.array(quest))
    ugc_hits_arr = _sanitize(np.array(ugc_hits))
    poss_hits_arr = _sanitize(np.array(poss_hits))
    sent_doc_arr = _sanitize(np.array(sent_doc))
    sent_std_arr = _sanitize(np.array(sent_std))

    denom = np.maximum(char_lens, 1.0)
    feat: dict[str, np.ndarray] = {
        "char_len": char_lens,
        "word_count": word_counts,
        "n_sentences": n_sents_arr,
        "mean_sentence_char_len": mean_sent_len_arr,
        "exclaim_per_1k": _sanitize(excl_arr / denom * 1000),
        "question_per_1k": _sanitize(quest_arr / denom * 1000),
        "lex_ugc_hits": ugc_hits_arr,
        "lex_poss_hits": poss_hits_arr,
        "lex_ugc_density": _sanitize(ugc_hits_arr / np.maximum(word_counts, 1.0)),
        "lex_poss_density": _sanitize(poss_hits_arr / np.maximum(word_counts, 1.0)),
        "sentiment_snownlp": sent_doc_arr,
        "sentiment_sent_std": sent_std_arr,
    }

    base_keys = list(feat.keys())

    vocab_meta: dict = {"char_tfidf": [], "word_tfidf": [], "svd_topics": []}

    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=128,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    X_char = char_vec.fit_transform(texts)
    char_names = char_vec.get_feature_names_out()
    char_keys: list[str] = []
    for j in range(X_char.shape[1]):
        name = f"char_tfidf_{j}"
        char_keys.append(name)
        feat[name] = _sanitize(np.asarray(X_char[:, j].todense()).ravel())
        vocab_meta["char_tfidf"].append(
            {"col": name, "ngram": char_names[j]}
        )

    word_vec = TfidfVectorizer(
        analyzer=zh_word_unigram_bigram,
        max_features=96,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    X_word = word_vec.fit_transform(texts)
    word_names = word_vec.get_feature_names_out()
    word_keys: list[str] = []
    for j in range(X_word.shape[1]):
        name = f"word_tfidf_{j}"
        word_keys.append(name)
        feat[name] = _sanitize(np.asarray(X_word[:, j].todense()).ravel())
        vocab_meta["word_tfidf"].append(
            {"col": name, "token": word_names[j]}
        )

    X_stack = sparse.hstack([X_char, X_word], format="csr")
    n_components = min(32, X_stack.shape[1] - 1, max(2, n // 3))
    lsa_keys: list[str] = []
    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_svd = svd.fit_transform(X_stack)
        for k in range(n_components):
            name = f"lsa_{k}"
            lsa_keys.append(name)
            feat[name] = _sanitize(X_svd[:, k])
        vocab_meta["svd_topics"] = {
            "n_components": n_components,
            "explained_variance_ratio_sum": float(
                np.sum(svd.explained_variance_ratio_)
            ),
        }

    feat["label_tag"] = _sanitize(
        np.array(
            [int(x) if str(x).isdigit() else 1 for x in tags], dtype=np.float64
        )
    )

    ordered_keys: list[str] = (
        base_keys + char_keys + word_keys + lsa_keys
    )
    return feat, vocab_meta, ordered_keys


def _open_csv_out(path: Path):
    try:
        f = path.open("w", encoding="utf-8-sig", newline="")
        return path, f
    except PermissionError:
        alt = path.with_name(path.stem + "_run" + path.suffix)
        print(f"注意：{path.name} 被占用，已写入 {alt.name}")
        return alt, alt.open("w", encoding="utf-8-sig", newline="")


def main() -> None:
    import csv

    FEATURES.mkdir(parents=True, exist_ok=True)
    merged = PROCESSED / "merged.csv"
    rows: list[tuple[str, str, str, str]] = []
    with merged.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append((r["book"], r["chapter"], r["paragragh"], r["tag"]))

    texts = [p for _, _, p, _ in rows]
    tags = [t for _, _, _, t in rows]
    feat_dict, vocab_meta, ordered_keys = build_features(texts, tags)

    vocab_path = FEATURES / "NLP_Feature_vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, ensure_ascii=False, indent=2)

    # ---------- 完整表（含章节名）----------
    out_path = FEATURES / "NLP_Feature_Matrix.csv"
    colnames = ["idx", "book", "chapter"] + ordered_keys + ["label_tag"]
    path_written, f_out = _open_csv_out(out_path)
    with f_out as f:
        w = csv.writer(f)
        w.writerow(colnames)
        for i, (bk, ch, _, _) in enumerate(rows):
            row = [i, bk, ch]
            for k in ordered_keys:
                row.append(feat_dict[k][i])
            row.append(feat_dict["label_tag"][i])
            w.writerow(row)

    # ---------- 仅数值：书名 one-hot + 全部特征 + label（适合 sklearn）----------
    book_names = sorted({r[0] for r in rows})
    n = len(rows)
    K = len(book_names)
    ohe = np.zeros((n, K), dtype=np.float64)
    for i, r in enumerate(rows):
        j = book_names.index(r[0])
        ohe[i, j] = 1.0

    train_cols = (
        [f"book_{b}" for b in book_names]
        + ordered_keys
        + ["label_tag"]
    )
    train_path = FEATURES / "NLP_Feature_train.csv"
    path_train, f_train = _open_csv_out(train_path)
    with f_train as f:
        w = csv.writer(f)
        w.writerow(train_cols)
        for i in range(n):
            row = list(ohe[i]) + [feat_dict[k][i] for k in ordered_keys]
            row.append(feat_dict["label_tag"][i])
            w.writerow(row)

    # ---------- StandardScaler：仅对连续特征缩放，书名 one-hot 保持 0/1（供线性 SVM）----------
    X_feat_only = np.column_stack([feat_dict[k] for k in ordered_keys])
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_feat_scaled = scaler.fit_transform(X_feat_only)
    X_feat_scaled = np.nan_to_num(
        X_feat_scaled, nan=0.0, posinf=0.0, neginf=0.0
    )
    X_scaled = np.hstack([ohe, X_feat_scaled])

    scaled_path = FEATURES / "NLP_Feature_train_scaled.csv"
    path_scaled, f_scaled = _open_csv_out(scaled_path)
    with f_scaled as f:
        w = csv.writer(f)
        w.writerow(train_cols)
        for i in range(n):
            row = list(X_scaled[i]) + [feat_dict["label_tag"][i]]
            w.writerow(row)

    meta_cols = {
        "book_onehot_order": book_names,
        "feature_order_after_oh": ordered_keys,
        "label_column": "label_tag",
        "n_samples": n,
        "n_features_X": len(train_cols) - 1,
        "note": "NLP_Feature_train.csv 未缩放，适合 RandomForest；NLP_Feature_train_scaled.csv 已 StandardScaler，适合线性 SVM。",
    }
    with (FEATURES / "NLP_Feature_train_columns.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(meta_cols, f, ensure_ascii=False, indent=2)

    print(
        f"已写入 {path_written.name}、{path_train.name}、{path_scaled.name}，"
        f"样本 {n}，训练特征维 {len(train_cols) - 1}（含书名 one-hot）"
    )


if __name__ == "__main__":
    main()
