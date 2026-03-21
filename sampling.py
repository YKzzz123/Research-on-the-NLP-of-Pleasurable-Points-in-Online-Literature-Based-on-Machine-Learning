import pandas as pd

def generate_annotation_samples(input_csv, output_csv, sample_size=1000):
    print("🚀 正在读取切分好的段落数据...")
    df = pd.read_csv(input_csv)
    
    print("🧱 正在构建上下文滑动窗口 (前两段 + 当前段)...")
    # 拼接上下文，为团队打标签提供完美的阅读语境
    df['Context_Text'] = (
        "【前文回顾】: " + df['Text'].shift(2).fillna('') + "\n" + 
        "【前文回顾】: " + df['Text'].shift(1).fillna('') + "\n" + 
        "【当前段落】: " + df['Text']
    )

    # ==========================================
    # 🌟 我们刚刚结合算法与人工提纯的《元尊》黄金爽点词库
    # ==========================================
    climax_keywords = [
        '震撼', '竟然', '忍不住', '可怕', '恐怖', 
        '爆发', '突破', '极限', '响彻', '暴射', '磅礴',
        '淘汰', '成功', '贯穿'
    ]
    print(f"🎯 正在使用 {len(climax_keywords)} 个核心情绪词进行定向探测...")
    
    # 寻找包含关键词的潜在高潮段落
    climax_mask = df['Context_Text'].str.contains('|'.join(climax_keywords), na=False)

    potential_climax_df = df[climax_mask]
    normal_df = df[~climax_mask] 

    print(f"📊 探测结果：发现潜在高潮段落 {len(potential_climax_df)} 个，普通段落 {len(normal_df)} 个。")
    
    # 分层抽样 (保证给团队标的数据里，爽点和非爽点比例均衡)
    half_size = sample_size // 2
    # 防御性编程：两类段落都不超过各自可用数量
    actual_climax_samples = min(half_size, len(potential_climax_df))
    actual_normal_samples = min(sample_size - actual_climax_samples, len(normal_df))
    
    print(f"🎲 正在随机抽取 {actual_climax_samples} 个疑似爽点 和 {actual_normal_samples} 个普通段落...")
    sample_climax = potential_climax_df.sample(n=actual_climax_samples, random_state=42)
    sample_normal = normal_df.sample(n=actual_normal_samples, random_state=42)

    # 合并、打乱顺序
    final_df = pd.concat([sample_climax, sample_normal]).sample(frac=1, random_state=42)
    
    # 新增一列留空，专门给团队填 0 或 1
    final_df['Is_Climax'] = '' 
    
    # 只保留必需的列输出
    output_df = final_df[['Chunk_ID', 'Context_Text', 'Is_Climax']]
    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print("-" * 40)
    print(f"🎉 抽样完成！已生成包含 {len(output_df)} 条数据的最终标注任务表：{output_csv}")
    print("👉 任务分配建议：将表格分成几份发给团队，大家只需在 Is_Climax 列填 1 (爽) 或 0 (平淡)。")

# --- 运行执行 ---
if __name__ == "__main__":
    # 输入：第一步切分好的 chunks 表格
    # 输出：发给团队的最终标注表格
    generate_annotation_samples('yuanzun_step1_chunks.csv', 'yuanzun_team_annotation_task.csv')