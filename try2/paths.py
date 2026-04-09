# -*- coding: utf-8 -*-
"""try2 目录约定：数据、模型与各类输出路径（脚本从此模块引用，避免硬编码）。"""
from pathlib import Path

TRY2 = Path(__file__).resolve().parent

DATA = TRY2 / "data"
RAW_ANNOTATIONS = DATA / "raw_annotations"
ANNOTATIONS_CSV = DATA / "annotations_csv"
PROCESSED = DATA / "processed"
FEATURES = DATA / "features"

ARTIFACTS = TRY2 / "artifacts"

OUTPUTS = TRY2 / "outputs"
TRAINING_FIGURES = OUTPUTS / "training_figures"
EVALUATION = OUTPUTS / "evaluation"
CV_RESULTS = OUTPUTS / "cv_results"
FEATURE_SELECTION = OUTPUTS / "feature_selection"
DATASET_VISUALIZATIONS = OUTPUTS / "dataset_visualizations"

DOCS = TRY2 / "docs"
