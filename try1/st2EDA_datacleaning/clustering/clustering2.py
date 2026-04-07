import pandas as pd
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# ⚠️ 解决 Matplotlib 中文乱码（请根据系统调整）
# Windows 用户通常用 'SimHei'，Mac 用户请改为 'Arial Unicode MS' 或 'PingFang HK'
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
# ==========================================

def discover_clean_emotion_keywords(csv_file_path, num_clusters=5, top_n_words=15):
    print("🚀 开始执行：终极纯净版聚类分析...")
    df = pd.read_csv(csv_file_path)
    texts = df['Text'].dropna().tolist()
    
    print("✂️ 正在进行最严格的【动词/形容词】分词过滤...")
    corpus = []
    
    # 🌟 终极停用词表：精准狙击所有高频的“语法废话”和“状态词”
    # 保留了：爆发、恐怖、狠狠、颤抖、骇然、不可思议 等真正有情绪张力的词
    stopwords = {
        '这个', '这么', '那么', '什么', '怎么', '自己', '知道', '没有', '就是', '有些', 
        '开始', '出现', '发现', '觉得', '感到', '说道', '起来', '有着', '却是', '直接', 
        '想要', '便是', '出来', '只见', '就算', '不会', '可能', '毕竟', '似乎', '变得', 
        '迅速', '终于', '能够', '应该', '面对', '正是', '最终', '真的', '真是', '的确',
        '宛如', '化为', '顿时', '渐渐', '响起', '涌现', '仿佛', '形成', '见到', '离开', 
        '显然', '之中', '那些', '一个', '出手', '清澈', '微微', '此时', '然后', '不过'
    }
    
    for text in texts:
        words = pseg.cut(text)
        valid_words = []
        for word, flag in words:
            # 严格过滤：词长必须大于1，不在黑名单中，且必须是动词(v)、形容词(a)、副词(d)、叹词(e)
            if len(word) > 1 and word not in stopwords:
                if flag[0] in ['v', 'a', 'd'] or flag in ['e', 'y']:
                    valid_words.append(word)
        corpus.append(" ".join(valid_words))

    print("🧮 正在计算 TF-IDF 权重矩阵...")
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    print(f"🤖 正在执行 K-Means 聚类 (设定分为 {num_clusters} 类)...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    print("\n🎉 === 剔除噪音后的纯净套路揭秘 === 🎉\n")
    feature_names = vectorizer.get_feature_names_out()
    centroids = kmeans.cluster_centers_
    
    for i in range(num_clusters):
        top_indices = centroids[i].argsort()[-top_n_words:][::-1]
        top_words = [feature_names[index] for index in top_indices]
        print(f"🔥 【纯净情绪流派 {i+1}】的核心词:")
        print("  |  ".join(top_words))
        print("-" * 60)

    # --- 🎨 重新生成词云图 ---
    print("\n🎨 正在生成纯净版词云图...")
    fig_wc, axes = plt.subplots(1, num_clusters, figsize=(4 * num_clusters, 4))
    if num_clusters == 1: axes = [axes]
    
    for i in range(num_clusters):
        top_indices = centroids[i].argsort()[-top_n_words:][::-1]
        word_freq = {feature_names[index]: centroids[i][index] for index in top_indices}
        
        # Windows 用户建议手动加 font_path='C:/Windows/Fonts/simhei.ttf' 防止词云方块字
        # Mac 用户建议加 font_path='/System/Library/Fonts/PingFang.ttc'
        try:
            wc = WordCloud(font_path='C:/Windows/Fonts/simhei.ttf', width=600, height=600, background_color='white', colormap='Reds').generate_from_frequencies(word_freq)
            axes[i].imshow(wc, interpolation='bilinear')
        except Exception:
            axes[i].text(0.5, 0.5, "请配置词云字体路径", ha='center', va='center')
        
        axes[i].set_title(f'情绪流派 {i+1}', fontsize=14, fontweight='bold')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('clean_wordclouds.png', dpi=300)
    print("✅ 纯净版词云已保存为 'clean_wordclouds.png'")
    plt.show()

# --- 运行执行 ---
if __name__ == "__main__":
    INPUT_CSV = 'yuanzun_step1_chunks.csv' 
    discover_clean_emotion_keywords(INPUT_CSV, num_clusters=4, top_n_words=12)