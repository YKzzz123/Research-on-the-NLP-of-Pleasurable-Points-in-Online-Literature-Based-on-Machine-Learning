import pandas as pd
import os

def batch_generate_master_task(input_csv_list, output_master_csv, sample_per_book=300):
    print(f"🚀 启动自动化流水线：目标总数据量 {sample_per_book * len(input_csv_list)} 条...")
    
    # 作者专属黄金爽点词库（全书通用）
    climax_keywords = [
        '震撼', '竟然', '忍不住', '可怕', '恐怖', 
        '爆发', '突破', '极限', '响彻', '暴射', '磅礴',
        '淘汰', '成功', '贯穿'
    ]
    
    all_sampled_data = [] 
    
    for csv_file in input_csv_list:
        if not os.path.exists(csv_file):
            print(f"⚠️ 警告：找不到文件 {csv_file}，跳过处理。")
            continue
            
        print(f"\n📖 正在处理: {csv_file}")
        df = pd.read_csv(csv_file)
        book_name = csv_file.split('_')[0] 
        
        # 1. 构建包含 3 个 Chunk 的上下文窗口 (200字/Chunk，总阅读量约600字)
        df['Context_Text'] = (
            "【前文回顾】: " + df['Text'].shift(2).fillna('') + "\n" + 
            "【前文回顾】: " + df['Text'].shift(1).fillna('') + "\n" + 
            "【当前段落】: " + df['Text']
        )
        
        # 标明书名
        df['Context_Text'] = f"[{book_name.upper()}] \n" + df['Context_Text']
        
        # 2. 定向探测
        climax_mask = df['Context_Text'].str.contains('|'.join(climax_keywords), na=False)
        potential_climax_df = df[climax_mask]
        normal_df = df[~climax_mask] 
        
        # 3. 分层抽样：目标各 half_size，不足时互相补位，最终不超过各自总量
        half_size = sample_per_book // 2
        actual_climax_samples = min(half_size, len(potential_climax_df))
        actual_normal_samples = min(sample_per_book - actual_climax_samples, len(normal_df))
        # 如果 normal 也不够，把空余额度还给 climax
        if actual_climax_samples + actual_normal_samples < sample_per_book:
            actual_climax_samples = min(sample_per_book - actual_normal_samples, len(potential_climax_df))
        
        sample_climax = potential_climax_df.sample(n=actual_climax_samples, random_state=42)
        sample_normal = normal_df.sample(n=actual_normal_samples, random_state=42)
        
        all_sampled_data.append(sample_climax)
        all_sampled_data.append(sample_normal)
        print(f"  ✅ 成功抽取 {len(sample_climax) + len(sample_normal)} 条数据。")

    # ==========================================
    # 🌟 终极合并与洗牌 (Shuffle)
    # ==========================================
    print("\n🌪️ 正在合并所有书籍数据，并进行终极全局大洗牌...")
    master_df = pd.concat(all_sampled_data)
    
    # 全局打乱顺序
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)
    master_df['Is_Climax'] = '' 
    
    final_output = master_df[['Chunk_ID', 'Context_Text', 'Is_Climax']]
    final_output.to_csv(output_master_csv, index=False, encoding='utf-8-sig')
    
    print("=" * 50)
    print(f"🎉 共合并 {len(input_csv_list)} 本书，生成了包含 {len(final_output)} 条数据的终极任务表！")
    print(f"📁 文件已保存为: {output_master_csv}")

# --- 运行执行 ---
if __name__ == "__main__":
    # 确保这 5 个基础文件都已经通过 step1_chunking.py (设置200字) 生成完毕
    my_books = [
        'row_chunks/yuanzun_chunks.csv',
        'row_chunks/doupo_chunks.csv',
        'row_chunks/wudong_chunks.csv',
        'row_chunks/dazhuzai_chunks.csv',
        'row_chunks/wanxiang_chunks.csv'
    ]
    
    # 一键生成 1500 条终极总表
    batch_generate_master_task(my_books, 'Team_Annotation_Task.csv', sample_per_book=300)