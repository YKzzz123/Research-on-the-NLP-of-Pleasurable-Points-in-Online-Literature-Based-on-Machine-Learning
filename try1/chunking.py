import pandas as pd
import re
import os

def clean_and_smart_chunking(txt_file_path, output_csv_path, min_chunk_length=200):
    print(f"🧹 正在读取并清洗文本: {txt_file_path} ...")
    
    with open(txt_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    raw_paragraphs = re.split(r'\n+', content)
    
    data_rows = []
    current_chunk = ""
    chunk_id = 1
    
    # ==========================================
    # 🛡️ 定义清洗规则 (Data Cleaning Rules)
    # ==========================================
    # 1. 定义需要整行过滤的“牛皮癣”废话（可根据你的txt情况自行添加）
    spam_keywords = ['求推荐', '求月票', '求打赏', '未完待续', '手机版阅读', '手机用户', '最新章节', '本站网址']
    
    # 2. 匹配章节名的正则表达式 (匹配开头是“第”，中间是数字或中文数字，跟着“章/卷”的行)
    chapter_pattern = re.compile(r'^第[零一二三四五六七八九十百千万0-9]+[章回节卷]')
    
    for p in raw_paragraphs:
        p = p.strip()
        
        # 🚨 第一道网：拦截章节标题
        if chapter_pattern.match(p):
            continue
            
        # 🚨 第二道网：拦截广告和求票废话
        if any(spam in p for spam in spam_keywords):
            continue
            
        # 🚨 第三道网：符号与乱码清洗 (就地修改)
        # 去除可能存在的 http 网址
        p = re.sub(r'http[s]?://\S+', '', p)
        # 去除无意义的连续特殊符号，比如 ***, ===, ---
        p = re.sub(r'[-=*~]{3,}', '', p)
        # 彻底清除段落中间的空格和换行符残余
        p = re.sub(r'\s+', '', p)
        
        # 过滤掉清洗后剩下不到5个字的极短行（通常是“嗯。”、“嘶。”）
        if len(p) < 5: 
            continue
            
        # ==========================================
        # 🧱 拼装逻辑块
        # ==========================================
        current_chunk += p + " " 
        
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
    print(f"✅ 清洗与分块双管齐下！成功生成了 {len(df)} 个纯净的逻辑段落。")
    print(f"📁 文件已保存为: {output_csv_path}")

# --- 运行执行 ---
if __name__ == "__main__":
    # 假设你依然在处理第一本书
    clean_and_smart_chunking('raw_books/wanxiang.txt', 'row_chunks/wanxiang_chunks.csv', min_chunk_length=200)
    clean_and_smart_chunking('raw_books/yuanzun.txt', 'row_chunks/yuanzun_chunks.csv', min_chunk_length=200)
    clean_and_smart_chunking('raw_books/dazhuzai.txt', 'row_chunks/dazhuzai_chunks.csv', min_chunk_length=200)
    clean_and_smart_chunking('raw_books/wudong.txt', 'row_chunks/wudong_chunks.csv', min_chunk_length=200)
    clean_and_smart_chunking('raw_books/doupo.txt', 'row_chunks/doupo_chunks.csv', min_chunk_length=200)
