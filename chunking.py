import pandas as pd
import re
import os

def smart_chunking(txt_file_path, output_csv_path, min_chunk_length=150):
    print(f"正在读取并智能合并文本: {txt_file_path} ...")
    
    with open(txt_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # 按回车切分原始物理行
    raw_paragraphs = re.split(r'\n+', content)
    
    data_rows = []
    current_chunk = ""
    chunk_id = 1
    
    for p in raw_paragraphs:
        p = p.strip()
        p = re.sub(r'\s+', '', p) 
        
        # 过滤极短废话
        if len(p) < 5: 
            continue
            
        current_chunk += p + " " 
        
        # 达到 150 字阈值，打包成一个块
        if len(current_chunk) >= min_chunk_length:
            data_rows.append({
                'Chunk_ID': f"C{chunk_id:05d}",
                'Text': current_chunk.strip()
            })
            chunk_id += 1
            current_chunk = "" 
            
    # 处理最后的尾巴
    if len(current_chunk) > 30:
        data_rows.append({
            'Chunk_ID': f"C{chunk_id:05d}",
            'Text': current_chunk.strip()
        })

    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 第一步大功告成！生成了 {len(df)} 个逻辑段落。文件保存为: {output_csv_path}")

# --- 运行执行 ---
if __name__ == "__main__":
    # 确保 raw_books 文件夹和 yuanzun.txt 存在
    smart_chunking('raw_books/yuanzun.txt', 'yuanzun_step1_chunks.csv')