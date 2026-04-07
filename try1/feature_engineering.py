import pandas as pd
import numpy as np
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

def run_feature_engineering_v2(excel_file, output_csv):
    print(f"🚀 启动特征工程 V2 (破除领域偏移)...")
    raw_df = pd.read_excel(excel_file)
    raw_df = raw_df.dropna(subset=['Is_Climax'])
    raw_df['Is_Climax'] = raw_df['Is_Climax'].astype(int)

    # 准备特征容器
    intensity_deltas = []
    exclamation_ratios = []
    current_texts_for_tfidf = []

    print("🧠 正在计算【动作/情绪密度反差】与【感叹号特征】...")
    for _, row in raw_df.iterrows():
        full_context = str(row['Context_Text'])
        
        # 分隔前文与当前段
        if "【当前段落】:" in full_context:
            parts = full_context.split("【当前段落】:")
            context_part = parts[0].replace("【前文回顾】:", "").strip()
            current_part = parts[1].strip()
        else:
            context_part = full_context 
            current_part = full_context

        # --------------------------------------------------
        # 🌟 核心替换：抛弃 SnowNLP，计算自定义的情绪动作密度
        # --------------------------------------------------
        def get_intensity_ratio(text):
            words = pseg.cut(text)
            hits, total = 0, 0
            for w, f in words:
                if len(w) > 1:
                    total += 1
                    if f[0] in ['v', 'a', 'd', 'e']: # 动词、形容词、副词、叹词
                        hits += 1
            return (hits / total) if total > 0 else 0

        context_intensity = get_intensity_ratio(context_part)
        current_intensity = get_intensity_ratio(current_part)
        
        # 计算密度反差：当前段落的激烈程度 - 前文的激烈程度
        intensity_deltas.append(current_intensity - context_intensity)

        # --------------------------------------------------
        # 🌟 新增特征：感叹号密度 (Exclamation Ratio)
        # --------------------------------------------------
        exc_count = current_part.count('！') + current_part.count('!')
        char_count = len(current_part) if len(current_part) > 0 else 1
        exclamation_ratios.append(exc_count / char_count)

        # 提取当前段落的有效词汇，喂给 TF-IDF
        clean_words = [w for w, f in pseg.cut(current_part) if len(w) > 1 and f[0] in ['v', 'a', 'd', 'e']]
        current_texts_for_tfidf.append(" ".join(clean_words))

    # --------------------------------------------------
    # 🌟 解除封印：让 TF-IDF 自由提取 Top 150 真实高频特征
    # --------------------------------------------------
    print("🧮 正在执行自由生长的 TF-IDF 提取...")
    # max_features=150: 提取最重要的 150 个词汇作为维度
    # min_df=3: 这个词至少要在 3 个段落里出现过，过滤极其冷门的错别字
    vectorizer = TfidfVectorizer(max_features=150, min_df=3)
    tfidf_matrix = vectorizer.fit_transform(current_texts_for_tfidf)
    
    # 获取机器自己找到的 150 个词
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"TF_{w}" for w in feature_names])

    # 拼接所有特征
    final_df = raw_df[['Chunk_ID', 'Is_Climax']].copy()
    final_df['Feat_Intensity_Delta'] = intensity_deltas
    final_df['Feat_Exclamation_Ratio'] = exclamation_ratios
    final_df = pd.concat([final_df, tfidf_df], axis=1)

    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("=" * 50)
    print(f"🎉 V2 特征矩阵已生成: {output_csv}")
    print(f"提取出的真实维度包括：{list(feature_names[:10])} ... 等 {len(feature_names)} 个词")

if __name__ == "__main__":
    run_feature_engineering_v2('Team_Annotated.xlsx', 'Final_Feature_Matrix_V2.csv')