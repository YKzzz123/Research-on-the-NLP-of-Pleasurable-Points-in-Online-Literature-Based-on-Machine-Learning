import pandas as pd

df = pd.read_csv('Final_Feature_Matrix.csv')

# 分离爽点和普通段落
climax_df = df[df['Is_Climax'] == 1]
normal_df = df[df['Is_Climax'] == 0]

print("🔍 爽点数据 (198条) 真实面貌大揭秘：")
print("-" * 50)

# 1. 检验 SnowNLP 是否失效
print(f"爽点段落平均情感反差值 (Delta): {climax_df['Feat_Delta'].mean():.4f}")
print(f"平淡段落平均情感反差值 (Delta): {normal_df['Feat_Delta'].mean():.4f}")
print("👉 如果这两个值差不多，甚至爽点的 Delta 更低，说明 SnowNLP 完全水土不服！")

# 2. 检验 TF-IDF 矩阵是否全军覆没 (零值灾难)
# 提取所有 TF_ 开头的列
tf_columns = [col for col in df.columns if col.startswith('TF_')]
climax_tf_sum = climax_df[tf_columns].sum(axis=1)

# 统计有多少个真实的爽点，连一个黄金词都没有包含（TFIDF全为0）
zero_feature_count = (climax_tf_sum == 0).sum()
print(f"\n⚠️ 198个爽点中，特征全为 0 的条数: {zero_feature_count}")
print(f"👉 如果这个数字很大（比如>50），说明14个词根本不够用，模型处于'盲人摸象'状态！")