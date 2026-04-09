#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from scipy import sparse
from text_features_common import zh_word_unigram_bigram

from paths import ARTIFACTS, EVALUATION, TRY2 as BASE

ART = ARTIFACTS
LABEL_NAME = {
    "tag1": "优越感 (Superiority)",
    "tag2": "占有感 (Acquisition Satisfaction)",
    "none": "无明显爽点 (No Strong Pleasure Signal)",
}
MODEL_NAME = {
    "vote": "Ensemble Voting",
    "svm": "SVM",
    "rf": "Random Forest",
    "nb": "Naive Bayes",
}
SAMPLE_TEXTS = {
    "Custom Input": "",
    "优越感示例 (Superiority Example)": (
        "“铛铛！” 然而，那些凝聚而成的元力光刃，在切割到林动手掌之上时，却是爆发出一道道火花，清脆的金铁之声迅传播开来，但最后，却不过只是在林动手臂上，留下了一道道白痕而已。 "
        "“这点本事，也敢出来丢人现眼！”林动冷笑"
        "，手掌猛然一拧，一股可怕的力量暴涌而出，只听得啪的一声，竟直接是在无数道震惊的目光中，生生地将华骨手中的骨矛捏爆而去。 "
        "“什么？” 见到坚硬无比的骨矛，居然被林动一把捏爆，那华骨也是骇得眼皮急跳，这才突然明白过来，眼前的林动，比起上次所见时，居然又是强悍了许多！"
    ),
    "占有感示例 (Acquisition Example)": (
        "“三兽蛮荒决！” 而望着这卷兽骨卷轴，萧炎忍不住的舔了舔嘴，难怪那两个老家伙见到自己的举动会如此的疯狂，原来是因为这个东西。"
    ),
    "无明显爽点示例 (Neutral Example)": (
        "天快黑了，街道两旁的店铺陆续关门。"
        "她买完晚饭后慢慢走回宿舍，路上只是想着明天还要继续整理资料。"
    ),
}
def get_theme(label: str) -> dict[str, str]:
    theme_map = {
        "tag1": {
            "accent": "#1f4a89",
            "soft": "#eaf2ff",
            "track": "#d9e6fb",
            "badge_text": "Superiority Signal",
            "hint": "This text leans toward the protagonist showing stronger superiority (优越感) through contrast and dominance.",
        },
        "tag2": {
            "accent": "#2e9f6b",
            "soft": "#eaf8f1",
            "track": "#d8f0e2",
            "badge_text": "Acquisition Signal",
            "hint": "This text leans toward acquisition satisfaction (占有感), where resources, rewards, or gains are obtained and possessed.",
        },
        "none": {
            "accent": "#6b7280",
            "soft": "#f3f4f6",
            "track": "#e5e7eb",
            "badge_text": "Neutral Signal",
            "hint": "The pleasure signal is relatively weak, so the text does not cross the threshold for a strong classification.",
        },
    }
    return theme_map.get(label, theme_map["none"])


def assess_input_text(text: str) -> tuple[list[str], list[str]]:
    stripped = text.strip()
    errors: list[str] = []
    warnings: list[str] = []
    if not stripped:
        errors.append("Please enter some text before running the prediction.")
        return errors, warnings

    meaningful = re.sub(r"[\W_]+", "", stripped, flags=re.UNICODE)
    if not meaningful:
        errors.append("The current input contains almost only whitespace or punctuation, so a meaningful prediction cannot be made.")
        return errors, warnings

    if len(meaningful) < 8:
        warnings.append("The text is quite short, so the model output may be unstable. At least 8 meaningful characters are recommended.")
    if len(stripped) > 600:
        warnings.append("The text is fairly long. The model will produce one overall judgment for the full passage.")
    unique_ratio = len(set(meaningful)) / max(len(meaningful), 1)
    if len(meaningful) >= 20 and unique_ratio < 0.35:
        warnings.append("The text contains a high level of repetition, which may weaken the model's ability to recognize key expressions.")
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", stripped):
        errors.append("The current input does not contain recognizable body text. Please check the content and try again.")
    return errors, warnings


def metric_card(title: str, value: str, subtitle: str = "") -> str:
    subtitle_html = f'<div class="metric-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    return f"""
    <div class="metric-card">
      <div class="metric-title">{html.escape(title)}</div>
      <div class="metric-value">{html.escape(value)}</div>
      {subtitle_html}
    </div>
    """


def history_panel(history: list[dict[str, str]]) -> str:
    if not history:
        return (
            '<div class="history-panel history-empty">'
            'No prediction history yet. After the first prediction, the most recent results will appear here for quick comparison.'
            "</div>"
        )

    rows = []
    for item in history[:8]:
        theme = get_theme(item["label_key"])
        rows.append(
            (
                '<div class="history-row">'
                '<div class="history-meta">'
                f'<span>{html.escape(item["time"])}</span>'
                f'<span class="history-badge" style="background:{theme["soft"]};color:{theme["accent"]};">'
                f'{html.escape(item["label"])}'
                "</span>"
                f'<span>{html.escape(item["model"])}</span>'
                f'<span>Score {html.escape(item["score"])}</span>'
                "</div>"
                f'<div class="history-snippet">{html.escape(item["snippet"])}</div>'
                "</div>"
            )
        )
    return f'<div class="history-panel">{"".join(rows)}</div>'


def push_history(history: list[dict[str, str]], out: dict[str, float | str], text: str) -> list[dict[str, str]]:
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "label": LABEL_NAME.get(str(out["final_label"]), str(out["final_label"])),
        "label_key": str(out["final_label"]),
        "model": MODEL_NAME["vote"],
        "score": f'{float(out["score"]):.2f}',
        "snippet": text.strip().replace("\n", " ")[:90],
    }
    return [entry, *history][:8]


def clear_input_box() -> None:
    st.session_state["input_text"] = ""
    st.session_state["pred_out"] = None
    st.session_state["last_loaded_sample"] = None
    st.session_state["sample_reset_pending"] = True


def artifact_signature() -> tuple[tuple[str, int], ...]:
    names = [
        "char_vec.joblib",
        "word_vec.joblib",
        "svm_model.joblib",
        "rf_model.joblib",
        "voting_model.joblib",
        "nb_model.joblib",
        "score_calibrator.joblib",
        "meta.json",
    ]
    signature = []
    for name in names:
        path = ART / name
        mtime = path.stat().st_mtime_ns if path.exists() else -1
        signature.append((name, mtime))
    return tuple(signature)

@st.cache_resource
def load_artifacts(_signature: tuple[tuple[str, int], ...]):
    char_vec = joblib.load(ART / "char_vec.joblib")
    word_vec = joblib.load(ART / "word_vec.joblib")
    svm = joblib.load(ART / "svm_model.joblib")
    rf = joblib.load(ART / "rf_model.joblib")
    voting = joblib.load(ART / "voting_model.joblib")
    nb = None
    score_calibrator = None
    nb_path = ART / "nb_model.joblib"
    if nb_path.exists():
        nb = joblib.load(nb_path)
    calibrator_path = ART / "score_calibrator.joblib"
    if calibrator_path.exists():
        score_calibrator = joblib.load(calibrator_path)
    with (ART / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return char_vec, word_vec, svm, rf, voting, nb, score_calibrator, meta


def load_model_scores():
    p = EVALUATION / "model_eval.csv"
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
    char_vec, word_vec, svm, rf, voting, nb, score_calibrator, meta = load_artifacts(artifact_signature())
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

    calibrated_conf = conf
    score_mode = "raw_vote_confidence"
    if score_calibrator is not None:
        calibrated_conf = float(score_calibrator.predict([conf])[0])
        score_mode = str(meta.get("score_mode", "calibrated_vote_confidence"))

    # 爽度分数（0~100）：优先使用校准后的 Voting 概率
    score = round(calibrated_conf * 100, 2)
    return {
        "final_label": label,
        "confidence": conf,
        "score": score,
        "score_mode": score_mode,
        "score_confidence": calibrated_conf,
        "p_svm_tag1": float(p_svm[0]),
        "p_svm_tag2": float(p_svm[1]),
        "p_rf_tag1": float(p_rf[0]),
        "p_rf_tag2": float(p_rf[1]),
        "p_vote_tag1": float(p_vote[0]),
        "p_vote_tag2": float(p_vote[1]),
        "p_nb_tag1": float(p_nb[0]),
        "p_nb_tag2": float(p_nb[1]),
        "has_nb": nb is not None,
        "selected_model": "vote",
        "selected_tag1": float(p_vote[0]),
        "selected_tag2": float(p_vote[1]),
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
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .card:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 28px rgba(18, 37, 63, 0.11);
        }
        .section-kicker {
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #5f7288;
        }
        .section-title {
            font-size: 20px;
            font-weight: 700;
            margin: 4px 0 6px 0;
        }
        .section-subtitle {
            font-size: 13px;
            color: #6b7280;
            margin-bottom: 10px;
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
            gap: 12px;
            margin-bottom: 14px;
        }
        .result-label {
            font-size: 23px;
            font-weight: 700;
            color: #0f2942;
        }
        .result-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 700;
            margin-right: 10px;
        }
        .result-pill.tag1 {
            background: #eaf2ff;
            color: #1f4a89;
        }
        .result-pill.tag2 {
            background: #eaf8f1;
            color: #2e9f6b;
        }
        .result-pill.none {
            background: #f3f4f6;
            color: #6b7280;
        }
        .result-headline {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }
        .result-hint {
            font-size: 14px;
            color: #4b5563;
            margin-top: 4px;
        }
        .confidence {
            font-size: 14px;
            color: #4b5563;
            background: #f3f6fb;
            padding: 6px 10px;
            border-radius: 999px;
        }
        .summary-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 10px 24px rgba(17, 37, 63, 0.08);
            margin-top: 10px;
            margin-bottom: 10px;
            border: 1px solid rgba(95, 114, 136, 0.14);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .summary-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 28px rgba(17, 37, 63, 0.12);
        }
        .metric-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 8px 20px rgba(17, 37, 63, 0.08);
            border: 1px solid rgba(95, 114, 136, 0.1);
            margin: 6px 0 10px 0;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .metric-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 24px rgba(17, 37, 63, 0.12);
        }
        .metric-title {
            font-size: 12px;
            font-weight: 700;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 800;
            color: #0f2942;
            margin-top: 8px;
        }
        .metric-sub {
            font-size: 13px;
            color: #6b7280;
            margin-top: 6px;
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
            background: conic-gradient(var(--ring-color, #5f7288) 0turn, var(--ring-color, #5f7288) var(--score), var(--ring-track, #dce4ef) var(--score), var(--ring-track, #dce4ef) 1turn);
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
            border: 1px solid rgba(95, 114, 136, 0.09);
            box-shadow: 0 8px 18px rgba(17, 37, 63, 0.05);
        }
        .prob-name {
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 10px;
            color: #1f2a37;
        }
        .input-status {
            border-radius: 14px;
            padding: 10px 12px;
            font-size: 13px;
            margin: 8px 0 10px 0;
        }
        .input-status.warning {
            background: #fff7ed;
            color: #9a3412;
            border: 1px solid #fed7aa;
        }
        .input-status.ok {
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
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
            transition: box-shadow 0.18s ease, transform 0.18s ease;
        }
        div[data-testid="stExpander"] details:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(17, 37, 63, 0.08);
        }
        div[data-testid="stTextArea"] textarea {
            background-color: #d1d5db !important;
            border: 1px solid #9ca3af !important;
            border-radius: 14px !important;
            box-shadow: inset 0 1px 4px rgba(55, 65, 81, 0.12) !important;
            color: #111827 !important;
            padding-top: 12px !important;
            padding-right: 34px !important;
        }
        div[data-testid="stTextArea"] [data-baseweb="base-input"] {
            background-color: #d1d5db !important;
            border-radius: 14px !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }
        div[data-testid="stTextArea"] [data-baseweb="base-input"]:focus-within {
            border-color: #5f7288 !important;
            box-shadow: 0 0 0 3px rgba(95, 114, 136, 0.16) !important;
        }
        .st-key-input_box {
            position: relative;
            background: #ffffff;
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 10px 24px rgba(17, 37, 63, 0.08);
            border: 1px solid rgba(95, 114, 136, 0.1);
            margin-bottom: 16px;
        }
        .st-key-input_box .st-key-clear_btn {
            position: absolute;
            top: 14px;
            right: 14px;
            z-index: 40;
            width: auto !important;
        }
        .st-key-input_box .st-key-clear_btn button {
            background: transparent !important;
            border: none !important;
            color: #374151 !important;
            box-shadow: none !important;
            min-height: 24px !important;
            height: 24px !important;
            padding: 0 6px !important;
            font-size: 16px !important;
            line-height: 1 !important;
            width: auto !important;
        }
        .st-key-clear_btn button:hover {
            background: rgba(255, 255, 255, 0.72) !important;
        }
        .st-key-predict_btn button,
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(90deg, #0e8f8d 0%, #0f766e 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
            min-height: 44px !important;
            padding: 0 20px !important;
            box-shadow: 0 10px 18px rgba(15, 118, 110, 0.24) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
        }
        .st-key-predict_btn button:hover {
            transform: translateY(-1px);
            filter: brightness(1.02);
            box-shadow: 0 14px 24px rgba(15, 118, 110, 0.3) !important;
        }
        .history-panel {
            background: #ffffff;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 10px 24px rgba(17, 37, 63, 0.08);
            border: 1px solid rgba(95, 114, 136, 0.1);
            margin: 10px 0 14px 0;
        }
        .history-row {
            padding: 12px 0;
            border-bottom: 1px solid #edf2f7;
        }
        .history-row:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }
        .history-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            font-size: 12px;
            color: #6b7280;
            margin-bottom: 6px;
        }
        .history-badge {
            border-radius: 999px;
            padding: 2px 8px;
            font-weight: 700;
        }
        .history-snippet {
            font-size: 14px;
            color: #1f2937;
            line-height: 1.6;
        }
        .history-empty {
            color: #6b7280;
            font-size: 14px;
        }
        .empty-result {
            background: #ffffff;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 10px 24px rgba(17, 37, 63, 0.06);
            color: #6b7280;
            border: 1px dashed #d7dee8;
            margin-top: 10px;
        }
        @media (max-width: 900px) {
            .top-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
            .score-top {
                flex-direction: column;
                align-items: flex-start;
            }
            .result-headline {
                align-items: flex-start;
            }
            .summary-card,
            .history-panel,
            .empty-result {
                padding: 16px;
            }
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
            <div class="header-title">网文爽感 (Web-Novel Pleasure) Classification & Scoring Demo</div>
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
    st.set_page_config(page_title="网文爽感 (Web-Novel Pleasure) Classification & Scoring Demo", page_icon="🔥", layout="wide")
    inject_style()
    render_header()

    if not ART.exists():
        st.error("Artifacts were not found. Please run: `python try2/train_text_models.py`")
        return

    _, _, _, _, _, _, _, meta = load_artifacts(artifact_signature())
    default_th = float(meta.get("unknown_threshold", 0.6))
    if "pred_out" not in st.session_state:
        st.session_state["pred_out"] = None
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = "那是神魄..."
    if "pred_history" not in st.session_state:
        st.session_state["pred_history"] = []
    if "sample_selector" not in st.session_state:
        st.session_state["sample_selector"] = "Custom Input"
    if "last_loaded_sample" not in st.session_state:
        st.session_state["last_loaded_sample"] = None
    if "sample_reset_pending" not in st.session_state:
        st.session_state["sample_reset_pending"] = False
    if st.session_state["sample_reset_pending"]:
        st.session_state["sample_selector"] = "Custom Input"
        st.session_state["sample_reset_pending"] = False

    with st.expander("⚙ Model Settings & Label Guide", expanded=False):
        threshold = st.slider(
            "none threshold (if the maximum probability is below this value, predict none)",
            0.3,
            0.9,
            default_th,
            0.01,
        )
        st.markdown(
            """
            **优越感 (Superiority)**  
            This label captures moments when the protagonist stands out through dominance, excellence, or contrast with others in wealth, status, strength, skill, or talent.  

            **占有感 (Acquisition Satisfaction)**  
            This label captures the pleasure of obtaining and possessing resources or rewards, such as treasures, techniques, equipment, experience, wealth, or companions.  
            """
        )

    st.markdown(
        """
        <div class="section-kicker">Input Workspace</div>
        <div class="section-title">Input Analysis Panel</div>
        <div class="section-subtitle">Use one-click samples and validate the input before prediction.</div>
        """,
        unsafe_allow_html=True,
    )
    selected_sample = st.radio(
        "Quick Sample Fill",
        list(SAMPLE_TEXTS.keys()),
        horizontal=True,
        key="sample_selector",
        label_visibility="collapsed",
    )
    st.caption("Switch between different text styles quickly to demonstrate how the model behaves across scenarios.")
    if selected_sample != "Custom Input" and st.session_state["last_loaded_sample"] != selected_sample:
        st.session_state["input_text"] = SAMPLE_TEXTS[selected_sample]
        st.session_state["pred_out"] = None
        st.session_state["last_loaded_sample"] = selected_sample
        st.rerun()
    if selected_sample == "Custom Input" and st.session_state["last_loaded_sample"] is not None:
        st.session_state["last_loaded_sample"] = None

    with st.container(key="input_box"):
        st.button("✕", key="clear_btn", help="Clear text", on_click=clear_input_box)
        st.text_area(
            "",
            key="input_text",
            height=220,
            label_visibility="collapsed",
            placeholder="Paste a web-novel excerpt (网文片段), or use one of the sample passages above.",
        )

    live_errors, live_warnings = assess_input_text(st.session_state["input_text"])
    if live_errors:
        st.markdown(
            f'<div class="input-status warning">{html.escape("；".join(live_errors))}</div>',
            unsafe_allow_html=True,
        )
    elif live_warnings:
        st.markdown(
            f'<div class="input-status warning">{html.escape("；".join(live_warnings))}</div>',
            unsafe_allow_html=True,
        )
    else:
        effective_len = len(re.sub(r"\s+", "", st.session_state["input_text"]))
        st.markdown(
            f'<div class="input-status ok">The current input looks good and is ready for prediction. Approximate effective character count: {effective_len}.</div>',
            unsafe_allow_html=True,
        )

    btn_l, btn_m, btn_r = st.columns([2.8, 2.4, 2.8])
    with btn_m:
        if st.button("Run Prediction", key="predict_btn", use_container_width=True, type="primary"):
            text = st.session_state["input_text"].strip()
            errors, warnings = assess_input_text(text)
            if errors:
                for msg in errors:
                    st.warning(msg)
            else:
                for msg in warnings:
                    st.info(msg)
                out = predict_with_none(text, threshold)
                st.session_state["pred_out"] = out
                st.session_state["pred_history"] = push_history(st.session_state["pred_history"], out, text)

    st.markdown(
        """
        <div class="section-kicker">Prediction Result</div>
        <div class="section-title">Result Display</div>
        <div class="section-subtitle">The results are organized into a summary, key metrics, score display, and detailed probabilities for easier explanation.</div>
        """,
        unsafe_allow_html=True,
    )
    out = st.session_state.get("pred_out")
    if out:
        theme = get_theme(out["final_label"])
        pretty_label = LABEL_NAME.get(out["final_label"], out["final_label"])
        score = float(out["score"])
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="section-kicker">Prediction Overview</div>
              <div class="score-top">
                <div>
                  <div class="result-headline">
                    <span class="result-pill {out["final_label"]}">{pretty_label}</span>
                    <div class="result-label">Prediction Ready</div>
                  </div>
                  <div class="result-hint">{theme["hint"]}</div>
                </div>
                <div class="confidence">⭐ Confidence: {out['confidence']:.4f}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card("Pleasure Score", f"{score:.2f}", "0 - 100"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("Active Model", MODEL_NAME["vote"], "Fixed to Ensemble Voting"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("Input Length", str(len(st.session_state["input_text"].strip())), "Counted from the current text"), unsafe_allow_html=True)
        with m4:
            st.markdown(metric_card("none Threshold", f"{threshold:.2f}", "Below this value, the result becomes none"), unsafe_allow_html=True)

        ring_col, insight_col = st.columns([1.05, 1.35])
        with ring_col:
            st.markdown(
                f"""
                <div class="score-card">
                  <div class="section-kicker">Score Gauge</div>
                  <div class="ring-wrap">
                    <div class="ring" style="--score:{score/100:.4f};--ring-color:{theme['accent']};--ring-track:{theme['track']};">
                      <div class="ring-inner">
                        <div class="ring-score">{score:.2f}</div>
                        <div class="ring-sub">Pleasure Score</div>
                      </div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with insight_col:
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="section-kicker">Model Insight</div>
                  <div class="detail-title">Decision Explanation</div>
                  <div class="result-hint">
                    The final decision is made by <b>{MODEL_NAME["vote"]}</b>. The model assigns
                    <b>{out["selected_tag1"] * 100:.2f}%</b> probability to “优越感 (Superiority)” and
                    <b>{out["selected_tag2"] * 100:.2f}%</b> probability to “占有感 (Acquisition Satisfaction)”.
                    The <b>{theme["badge_text"]}</b> is stronger, so this result is selected as the final classification.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📈 Detailed Probability Breakdown", expanded=True):
            model_specs = [
                ("svm", "SVM Probability", out["p_svm_tag1"], out["p_svm_tag2"]),
                ("rf", "Random Forest Probability", out["p_rf_tag1"], out["p_rf_tag2"]),
            ]
            if out.get("has_nb", False):
                model_specs.append(("nb", "Naive Bayes Probability", out["p_nb_tag1"], out["p_nb_tag2"]))
            model_specs.append(("vote", "Voting Probability", out["p_vote_tag1"], out["p_vote_tag2"]))

            cols = st.columns(len(model_specs))
            for col, (_, title, p1, p2) in zip(cols, model_specs):
                with col:
                    st.markdown(f'<div class="prob-col"><div class="prob-name">{title}</div>', unsafe_allow_html=True)
                    st.markdown(prob_row("优越感 (Superiority)", p1, "#1f4a89"), unsafe_allow_html=True)
                    st.markdown(prob_row("占有感 (Acquisition Satisfaction)", p2, "#2e9f6b"), unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="legend-row">
                  <span class="legend-dot" style="background:#1f4a89;"></span>优越感 (Superiority)
                  &nbsp;&nbsp;&nbsp;
                  <span class="legend-dot" style="background:#2e9f6b;"></span>占有感 (Acquisition Satisfaction)
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="empty-result">
              Enter some text and click “Run Prediction” to see the predicted label, pleasure score, model confidence, and detailed probability breakdown.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-kicker">Prediction History</div>
        <div class="section-title">Recent Predictions</div>
        <div class="section-subtitle">The latest 8 predictions are stored automatically for quick comparison across texts and settings.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(history_panel(st.session_state["pred_history"]), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
