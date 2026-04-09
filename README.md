# 项目说明
1. 首先，本项目是针对POLYU GAH MSc COMP5572课程的Project
2. 项目旨在仅使用传统机器学习，剖析网文为什么爽，爽在哪，并用NLP的方式对网文爽点进行分类并打分。
3. 可视化网站已完成部署，网址是：https://pleasurable-points-in-online-literature-classifier.streamlit.app/

# 文件夹说明
## 关于try1与try2文件夹的说明
本项目的数据集构建经历了两次不同方向的尝试，try1文件夹（废稿）下放置了第一次尝试的全过程，后证实构建方向错误，遂转向新的构建思路，相关操作放置在try2文件夹下，成功构建了有效的数据集，得到了最终的结果并将结果可视化。
## raw_books
该文件夹原本是为了try1流程服务的，其下文件如"yuanzun.txt"并非《元尊》原始文本，而是经过节选后的部分章节。其下row_text文件夹下才放置了五本小说的全部原始文本，命名如"yuanzun_all.txt"

# try2：爽感文本标注与分类实验
本目录包含从**标注合并 → NLP 特征 → 模型训练 → Streamlit 演示**的完整流水线。路径约定集中在 `paths.py`，脚本请从项目根目录用 `python try2/<脚本>.py` 调用（与原先一致）。

## 目录结构

| 路径 | 说明 |
|------|------|
| `*.py` | 流水线脚本与公共模块（`text_features_common.py`、`plot_utils.py`、`paths.py`） |
| `artifacts/` | 训练导出的模型与向量化器（`*.joblib`、`meta.json`） |
| `data/raw_annotations/` | 各书原始标注（`*.xlsx`、`wdqk.xls`） |
| `data/annotations_csv/` | 脚本生成的单书标注 CSV |
| `data/processed/` | 合并语料 `merged.csv` |
| `data/features/` | NLP 特征矩阵、词表与列说明 JSON、`selected_feature_list.json` |
| `outputs/training_figures/` | 训练过程图（分布、混淆矩阵、指标对比等） |
| `outputs/evaluation/` | 验证集评估表与报告（`model_eval*.csv`、`classification_report_table.*`） |
| `outputs/cv_results/` | 随机搜索交叉验证明细（`svm_cv_results.csv`、`rf_cv_results.csv`） |
| `outputs/feature_selection/` | 特征得分、冗余对与相关分析图 |
| `outputs/dataset_visualizations/` | 数据集探索性可视化（`merged/`、`selected_features/`） |
| `docs/` | 说明与交接类 Markdown |

## 推荐执行顺序

1. `python try2/merge_annotations.py` — 读原始标注，写出单书 CSV 与 `data/processed/merged.csv`
2. `python try2/build_nlp_features.py` — 生成 `data/features/` 下特征文件
3. `python try2/feature_selection_analysis.py`（可选）— 特征筛选与图表，写入 `outputs/feature_selection/` 与 `data/features/`
4. `python try2/train_text_models.py` — 训练并写入 `artifacts/`、`outputs/training_figures/`、`outputs/evaluation/`、`outputs/cv_results/`
5. `python try2/visualize_datasets.py`（可选）— 更新 `outputs/dataset_visualizations/`
6. `streamlit run try2/demo_app.py` — 交互演示（依赖 `artifacts/` 与 `outputs/evaluation/model_eval.csv`）
