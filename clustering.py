import pandas as pd
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def discover_emotion_keywords(csv_file_path, num_clusters=6, top_n_words=15):
    print("🔍 正在读取数据...")
    df = pd.read_csv(csv_file_path)
    texts = df['Text'].dropna().tolist()
    
    print("✂️ 正在进行【情绪导向】的深度分词过滤...")
    corpus = []
    
    # 扩大停用词表，去掉一些常见但无爽感的动词/副词
    stopwords = {'这个', '怎么', '知道', '没有', '就是', '有些', '开始', '出现', '发现', '觉得', '感到', '说道'}
    
    for text in texts:
        words = pseg.cut(text)
        valid_words = []
        for word, flag in words:
            # 核心改动：
            # 1. 彻底抛弃名词 (不包含 'n')
            # 2. 保留动词(v)、形容词(a)、副词(d)、叹词(e)、语气词(y)
            # 3. 词长必须大于 1（去掉单字动词如“走”、“看”，保留“震惊”、“暴退”等词）
            if len(word) > 1 and word not in stopwords:
                if flag[0] in ['v', 'a', 'd'] or flag in ['e', 'y']:
                    valid_words.append(word)
        corpus.append(" ".join(valid_words))

    print("🧮 正在计算 TF-IDF 权重矩阵 (聚焦动词与形容词)...")
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=5)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    print(f"🤖 正在执行 K-Means 聚类...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    
    print("\n🎉 === 聚焦情绪与动作的套路揭秘 === 🎉\n")
    feature_names = vectorizer.get_feature_names_out()
    centroids = kmeans.cluster_centers_
    
    for i in range(num_clusters):
        top_indices = centroids[i].argsort()[-top_n_words:][::-1]
        top_words = [feature_names[index] for index in top_indices]
        
        print(f"🔥 【动作/情绪流派 {i+1}】的核心词:")
        print("  |  ".join(top_words))
        print("-" * 60)

# --- 运行执行 ---
if __name__ == "__main__":
    INPUT_CSV = 'yuanzun_step1_chunks.csv' 
    discover_emotion_keywords(INPUT_CSV, num_clusters=5, top_n_words=15)