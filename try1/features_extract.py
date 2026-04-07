import pandas as pd
import numpy as np
import jieba.posseg as pseg
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

def run_feature_engineering(excel_file, output_csv):
    print(f"🚀 启动特征工程流水线...")
    
    # 1. 读取 Excel 并转化为中间 CSV 存档（按要求保留格式转换步骤）
    print(f"📂 正在读取标注文件: {excel_file}")
    raw_df = pd.read_excel(excel_file)
    
    # 清洗：确保标注列是整数且无空值
    raw_df = raw_df.dropna(subset=['Is_Climax'])
    raw_df['Is_Climax'] = raw_df['Is_Climax'].astype(int)
    
    # 存一份 CSV 备份，方便后续直接读取
    temp_csv = "temp_annotated_data.csv"
    raw_df.to_csv(temp_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 已将原始数据转化为 {temp_csv} 进行后续处理")

    # 2. 准备特征容器
    sentiment_deltas = []
    verb_adj_ratios = []
    current_texts_only = []

    print("🧠 正在提取深度特征（情感反差 + 词性密度）...")
    for _, row in raw_df.iterrows():
        full_context = str(row['Context_Text'])
        
        # 分隔前文与当前段
        if "【当前段落】:" in full_context:
            parts = full_context.split("【当前段落】:")
            context_part = parts[0].replace("【前文回顾】:", "").strip()
            current_part = parts[1].strip()
        else:
            context_part = full_context # 降级处理
            current_part = full_context

        # 计算情感反差 (Sentiment Delta)
        try:
            s_context = SnowNLP(context_part).sentiments if len(context_part) > 5 else 0.5
            s_current = SnowNLP(current_part).sentiments if len(current_part) > 5 else 0.5
            delta = s_current - s_context
        except:
            delta = 0.0
        sentiment_deltas.append(delta)

        # 计算动词/形容词占比
        words = pseg.cut(current_part)
        hits = 0
        total = 0
        clean_text_words = []
        for word, flag in words:
            if len(word) > 1:
                total += 1
                clean_text_words.append(word)
                if flag[0] in ['v', 'a', 'd', 'e']: # 动词、形容词、副词、叹词
                    hits += 1
        
        ratio = (hits / total) if total > 0 else 0
        verb_adj_ratios.append(ratio)
        current_texts_only.append(" ".join(clean_text_words))

    # 3. 定向词汇 TF-IDF 提取
    print("🧮 正在注入黄金词库 TF-IDF 权重...")
    gold_keywords = ['震撼', '竟然', '忍不住', '可怕', '恐怖', '爆发', '突破', '极限', '响彻', '暴射', '磅礴', '淘汰', '成功', '贯穿']
    vectorizer = TfidfVectorizer(vocabulary=gold_keywords)
    tfidf_matrix = vectorizer.fit_transform(current_texts_only)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"TF_{w}" for w in gold_keywords])

    # 4. 最终特征大拼接
    final_df = raw_df[['Chunk_ID', 'Is_Climax']].copy()
    final_df['Feat_Delta'] = sentiment_deltas
    final_df['Feat_Ratio'] = verb_adj_ratios
    final_df = pd.concat([final_df, tfidf_df], axis=1)

    # 保存最终矩阵
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("=" * 50)
    print(f"🎉 特征工程完成！最终训练矩阵已保存至: {output_csv}")
    print(f"📊 样本统计: 总数 {len(final_df)} | 爽点(1): {final_df['Is_Climax'].sum()} | 平淡(0): {len(final_df)-final_df['Is_Climax'].sum()}")

if __name__ == "__main__":
    run_feature_engineering('Team_Annotated.xlsx', 'Final_Feature_Matrix.csv')