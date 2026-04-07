import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# 解决 Matplotlib 中文显示问题 (请根据你的系统调整)
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 用 SimHei, Mac 用 Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False 

def train_and_evaluate_svm(feature_csv):
    print(f"🚀 启动终极训练引擎，读取特征矩阵: {feature_csv}")
    df = pd.read_csv(feature_csv)
    
    # 1. 分离特征矩阵 X 和 标签 y
    # 剔除掉无用的 ID 列和目标标签列，剩下的全是特征
    X = df.drop(columns=['Chunk_ID', 'Is_Climax'])
    y = df['Is_Climax']
    
    # 2. 划分训练集和测试集 (80% 训练, 20% 考试)
    # ⚠️ 极度关键参数: stratify=y 
    # 它可以保证在切分时，训练集和测试集里爽点(1)的比例都严格保持在 13% 左右，防止极端情况
    print("✂️ 正在切分数据集 (80% 训练集, 20% 测试集)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. 特征缩放 (Standardization) - SVM 的刚需
    # 我们的 Feat_Delta 是浮点数，TF-IDF 也是极小的浮点数，必须把它们缩放到同一个量级，SVM 才能跑得准
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 构建并训练 SVM 模型
    print("🧠 正在训练支持向量机 (SVM) 模型...")
    # ⚠️ 杀手锏参数: class_weight='balanced'
    # 使用 linear (线性核) 是因为文本特征通常线性可分，而且事后我们可以提取特征重要性！
    svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # 5. 在测试集上进行预测考试
    print("📝 模型训练完毕，正在 20% 的盲测数据集上进行考试...")
    y_pred = svm_model.predict(X_test_scaled)
    
    # ==========================================
    # 📊 生成专业评估报告与图表
    # ==========================================
    print("\n" + "="*50)
    print("🏆 【模型期末考试成绩单 (Classification Report)】")
    print("="*50)
    # 绝不能只看 Accuracy，重点看类别 1 的 Precision(精准率) 和 Recall(召回率)
    print(classification_report(y_test, y_pred, target_names=['平淡段落 (0)', '高潮爽点 (1)']))
    
    # 绘制混淆矩阵热力图
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['预测: 平淡(0)', '预测: 爽点(1)'],
                yticklabels=['真实: 平淡(0)', '真实: 爽点(1)'],
                annot_kws={"size": 16})
    plt.title('SVM 模型预测混淆矩阵 (Confusion Matrix)', fontsize=16, pad=20)
    plt.xlabel('AI 模型预测结果', fontsize=14)
    plt.ylabel('人类真实标注', fontsize=14)
    
    # 保存图片以供 Report 使用
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png', dpi=300)
    print("✅ 混淆矩阵图表已保存为 'svm_confusion_matrix.png'，请直接贴入你的 Final Report！")
    
    # 6. 探秘黑盒：什么特征最能决定“爽”？(权重分析)
    print("\n🔍 【模型决策揭秘：权重最高的 Top 5 爽感特征】")
    feature_names = X.columns
    coefs = svm_model.coef_[0]
    # 将特征名和对应的权重打包并按绝对值排序
    feature_importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    
    for feature, weight in feature_importance[:5]:
        direction = "⬆️ 极度增加爽感" if weight > 0 else "⬇️ 压抑爽感"
        print(f"  - {feature}: 权重 {weight:.4f} ({direction})")

    plt.show()

# --- 运行执行 ---
if __name__ == "__main__":
    train_and_evaluate_svm('Final_Feature_Matrix.csv')