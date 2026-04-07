#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from scipy import sparse
from text_features_common import zh_word_unigram_bigram

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
LABEL_NAME = {
    "tag1": "优越感",
    "tag2": "占有感",
    "none": "无明显爽点",
}

@st.cache_resource
def load_artifacts():
    char_vec = joblib.load(ART / "char_vec.joblib")
    word_vec = joblib.load(ART / "word_vec.joblib")
    svm = joblib.load(ART / "svm_model.joblib")
    rf = joblib.load(ART / "rf_model.joblib")
    voting = joblib.load(ART / "voting_model.joblib")
    nb = None
    nb_path = ART / "nb_model.joblib"
    if nb_path.exists():
        nb = joblib.load(nb_path)
    with (ART / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return char_vec, word_vec, svm, rf, voting, nb, meta


def load_model_scores():
    p = BASE / "model_eval.csv"
    if not p.exists():
        return None
    try:
        import pandas as pd

        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return None


def build_x(text: str, char_vec, word_vec):
    xc = char_vec.transform([text])
    xw = word_vec.transform([text])
    return sparse.hstack([xc, xw], format="csr")


def predict_with_none(text: str, threshold: float):
    char_vec, word_vec, svm, rf, voting, nb, meta = load_artifacts()
    x = build_x(text, char_vec, word_vec)
    xv = x.toarray()

    p_svm = svm.predict_proba(x)[0]
    p_rf = rf.predict_proba(xv)[0]
    if isinstance(voting, dict) and "w_svm" in voting and "w_rf" in voting:
        w_svm = float(voting["w_svm"])
        w_rf = float(voting["w_rf"])
        p_vote = w_svm * p_svm + w_rf * p_rf
    else:
        # 兼容旧版 artifacts（voting_model 为 sklearn VotingClassifier）
        p_vote = voting.predict_proba(xv)[0]
    p_nb = nb.predict_proba(x)[0] if nb is not None else np.array([0.0, 0.0], dtype=float)

    cls = int(np.argmax(p_vote) + 1)
    conf = float(np.max(p_vote))
    label = f"tag{cls}"
    if conf < threshold:
        label = "none"

    # 爽度分数（0~100）：用集成模型最大置信度映射
    score = round(conf * 100, 2)
    return {
        "final_label": label,
        "confidence": conf,
        "score": score,
        "p_svm_tag1": float(p_svm[0]),
        "p_svm_tag2": float(p_svm[1]),
        "p_rf_tag1": float(p_rf[0]),
        "p_rf_tag2": float(p_rf[1]),
        "p_vote_tag1": float(p_vote[0]),
        "p_vote_tag2": float(p_vote[1]),
        "p_nb_tag1": float(p_nb[0]),
        "p_nb_tag2": float(p_nb[1]),
        "has_nb": nb is not None,
        "meta_threshold": float(meta.get("unknown_threshold", 0.6)),
    }


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f2f4f8;
            color: #1f2937;
            font-family: "Inter", "PingFang SC", "Microsoft YaHei", sans-serif;
        }
        .top-header {
            background: #0f2942;
            color: #ffffff;
            border-radius: 14px;
            padding: 16px 22px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 22px rgba(15, 41, 66, 0.18);
        }
        .header-title {
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 0.3px;
        }
        .header-right {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .deploy-btn {
            background: #142f4a;
            border: 1px solid rgba(255,255,255,0.25);
            color: #ffffff;
            border-radius: 999px;
            padding: 7px 14px;
            font-size: 13px;
            font-weight: 600;
        }
        .icon-chip {
            width: 30px;
            height: 30px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.3);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
        }
        .card {
            background: #ffffff;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 8px 20px rgba(18, 37, 63, 0.08);
            margin-bottom: 14px;
        }
        .section-title {
            font-size: 20px;
            font-weight: 700;
            margin: 4px 0 10px 0;
        }
        .score-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 10px 24px rgba(17, 37, 63, 0.1);
            margin-top: 8px;
        }
        .score-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .result-label {
            font-size: 23px;
            font-weight: 700;
            color: #0f2942;
        }
        .confidence {
            font-size: 14px;
            color: #4b5563;
            background: #f3f6fb;
            padding: 6px 10px;
            border-radius: 999px;
        }
        .ring-wrap {
            display: flex;
            justify-content: center;
            margin: 8px 0 16px 0;
        }
        .ring {
            width: 170px;
            height: 170px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: conic-gradient(#5f7288 0turn, #5f7288 var(--score), #dce4ef var(--score), #dce4ef 1turn);
        }
        .ring-inner {
            width: 130px;
            height: 130px;
            border-radius: 50%;
            background: #ffffff;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .ring-score {
            font-size: 34px;
            line-height: 1;
            font-weight: 800;
            color: #0f2942;
        }
        .ring-sub {
            font-size: 12px;
            color: #6b7280;
            margin-top: 6px;
        }
        .detail-title {
            font-size: 18px;
            font-weight: 700;
            margin-top: 6px;
            margin-bottom: 8px;
        }
        .prob-col {
            background: #f8fafc;
            border-radius: 12px;
            padding: 12px;
        }
        .prob-name {
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 10px;
            color: #1f2a37;
        }
        .legend-row {
            margin-top: 10px;
            font-size: 13px;
            color: #4b5563;
        }
        .legend-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }
        div[data-testid="stExpander"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        div[data-testid="stExpander"] details {
            border: none !important;
            background: #ffffff;
            border-radius: 14px;
            padding: 2px 8px;
        }
        .input-wrap {
            position: relative;
        }
        .input-wrap div[data-testid="stTextArea"] textarea {
            background: #d1d5db !important;
            border: 1px solid #9ca3af !important;
            border-radius: 14px !important;
            box-shadow: inset 0 1px 4px rgba(55, 65, 81, 0.12) !important;
            color: #111827 !important;
            padding-top: 34px !important;
        }
        .input-wrap div[data-testid="stButton"] {
            position: absolute;
            right: -2px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 30;
        }
        .input-wrap div[data-testid="stButton"] button {
            background: transparent !important;
            border: none !important;
            color: #111111 !important;
            box-shadow: none !important;
            min-height: 24px !important;
            height: 24px !important;
            padding: 0 4px !important;
            font-size: 18px !important;
            line-height: 1 !important;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(90deg, #0e8f8d 0%, #0f766e 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
            min-height: 44px !important;
            box-shadow: 0 10px 18px rgba(15, 118, 110, 0.24) !important;
        }
        .score-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 13px;
        }
        .score-table th, .score-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #e5e7eb;
            text-align: right;
        }
        .score-table th:first-child, .score-table td:first-child {
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="top-header">
            <div class="header-title">网文爽感分类与评分 Demo</div>
            <div class="header-right">
                <div class="deploy-btn">Deploy</div>
                <div class="icon-chip">◧</div>
                <div class="icon-chip">☰</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prob_row(label: str, value: float, color: str) -> str:
    pct = max(0.0, min(100.0, value * 100))
    return f"""
    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px;">
        <span>{label}</span><span>{pct:.2f}%</span>
      </div>
      <div style="height:10px;background:#e5e7eb;border-radius:999px;overflow:hidden;">
        <div style="height:100%;width:{pct:.2f}%;background:{color};"></div>
      </div>
    </div>
    """


def main():
    st.set_page_config(page_title="网文爽感分类与评分 Demo", page_icon="🔥", layout="wide")
    inject_style()
    render_header()

    if not ART.exists():
        st.error("未找到 artifacts。请先运行：python try2/train_text_models.py")
        return

    _, _, _, _, _, _, meta = load_artifacts()
    default_th = float(meta.get("unknown_threshold", 0.6))
    if "pred_out" not in st.session_state:
        st.session_state["pred_out"] = None
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = "那是神魄..."

    with st.expander("⚙ 模型配置 & 标签说明", expanded=False):
        threshold = st.slider(
            "none 判定阈值（最大概率低于该值 => none）",
            0.3,
            0.9,
            default_th,
            0.01,
        )
        st.markdown(
            """
            **优越感（原 tag1）**  
            表现为众人瞩目之下主角的优势与卓异，常通过与配角在财富、身份、武力、技能、才华等方面对比，强化主角的独特与强大。  

            **占有感（原 tag2）**  
            表现为主角获得并占有资源与成果，如宝物、功法、装备、经验、钱财、伙伴等，核心是“获得”的满足与强化。  
            """
        )

    st.markdown('<div class="section-title">输入文本</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
    if st.button("✕", key="clear_btn", help="清空文本"):
        st.session_state["input_text"] = ""
        st.rerun()
    st.text_area("", key="input_text", height=220, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    btn_l, btn_m, btn_r = st.columns([3, 2, 3])
    with btn_m:
        if st.button("开始预测", use_container_width=True, type="primary"):
            t = st.session_state["input_text"].strip()
            if not t:
                st.warning("请输入文本。")
            else:
                st.session_state["pred_out"] = predict_with_none(t, threshold)

    out = st.session_state.get("pred_out")
    if out:
        pretty_label = LABEL_NAME.get(out["final_label"], out["final_label"])
        score = float(out["score"])
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="score-top">
              <div class="result-label">预测结果：{pretty_label}</div>
              <div class="confidence">⭐ 置信度：{out['confidence']:.4f}</div>
            </div>
            <div class="ring-wrap">
              <div class="ring" style="--score:{score/100:.4f};">
                <div class="ring-inner">
                  <div class="ring-score">{score:.2f}</div>
                  <div class="ring-sub">爽感得分</div>
                </div>
              </div>
            </div>
            <div class="detail-title">📈 详细概率分解</div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="prob-col"><div class="prob-name">SVM 概率</div>', unsafe_allow_html=True)
            st.markdown(prob_row("优越感", out["p_svm_tag1"], "#1f4a89"), unsafe_allow_html=True)
            st.markdown(prob_row("占有感", out["p_svm_tag2"], "#2e9f6b"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="prob-col"><div class="prob-name">随机森林概率</div>', unsafe_allow_html=True)
            st.markdown(prob_row("优越感", out["p_rf_tag1"], "#1f4a89"), unsafe_allow_html=True)
            st.markdown(prob_row("占有感", out["p_rf_tag2"], "#2e9f6b"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="prob-col"><div class="prob-name">Naive Bayes 概率</div>', unsafe_allow_html=True)
            st.markdown(prob_row("优越感", out["p_nb_tag1"], "#1f4a89"), unsafe_allow_html=True)
            st.markdown(prob_row("占有感", out["p_nb_tag2"], "#2e9f6b"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if not out.get("has_nb", False):
                st.caption("未检测到 nb_model.joblib")
        with c4:
            st.markdown('<div class="prob-col"><div class="prob-name">Voting 概率</div>', unsafe_allow_html=True)
            st.markdown(prob_row("优越感", out["p_vote_tag1"], "#1f4a89"), unsafe_allow_html=True)
            st.markdown(prob_row("占有感", out["p_vote_tag2"], "#2e9f6b"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="legend-row">
              <span class="legend-dot" style="background:#1f4a89;"></span>优越感
              &nbsp;&nbsp;&nbsp;
              <span class="legend-dot" style="background:#2e9f6b;"></span>占有感
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
