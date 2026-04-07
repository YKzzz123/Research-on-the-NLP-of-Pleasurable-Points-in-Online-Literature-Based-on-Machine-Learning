import os
import re
import pandas as pd

# 统计结果 CSV 输出目录（相对项目根目录，与 st1数据采集 同级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'st2EDA_datacleaning')


def analyze_chapter_lengths(txt_file_path):
    print(f"正在读取并分析文件: {txt_file_path} ...")
    
    with open(txt_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 核心：使用正则表达式匹配网文常见的章节标题
    # 匹配规则：以“第”开头，中间是中文数字或阿拉伯数字，跟着“章/回/节/卷”
    pattern = r'(?:^|\n)(第[零一二三四五六七八九十百千万0-9]+[章回节卷].*)'
    
    # 按章节标题对全文进行切分
    parts = re.split(pattern, content)
    
    data = []
    # parts 的结构：parts[0]通常是卷首语；parts[1]是第一章标题；parts[2]是第一章正文...以此类推
    for i in range(1, len(parts), 2):
        chapter_title = parts[i].strip()
        chapter_content = parts[i+1]
        
        # 清洗掉所有空格、换行符，只统计纯汉字/标点数量
        pure_text = re.sub(r'\s+', '', chapter_content)
        word_count = len(pure_text)
        
        # 过滤掉极个别可能是误判的超短章节（比如少于50字的）
        if word_count > 50:
            data.append({
                'Chapter_Title': chapter_title,
                'Word_Count': word_count
            })
        
    if not data:
        print("❌ 未能识别出章节标题，请检查源文件的章节名格式是否为'第X章'。")
        return
        
    df = pd.DataFrame(data)
    
    # --- 在终端输出统计报告 ---
    print("\n=== 📊 章节字数统计报告 ===")
    print(f"总计识别章节数: {len(df)} 章")
    print(f"最长章节字数: {df['Word_Count'].max()} 字")
    print(f"最短章节字数: {df['Word_Count'].min()} 字")
    print(f"平均章节字数: {df['Word_Count'].mean():.0f} 字")
    print(f"字数中位数: {df['Word_Count'].median():.0f} 字")
    print("==================================\n")
    
    # 导出详细表格 → st2EDA_datacleaning/<书名>_chapter_stats.csv
    base = os.path.splitext(os.path.basename(txt_file_path))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, f'{base}_chapter_stats.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 详细统计报表已生成: {output_csv}")

# --- 运行代码 ---
if __name__ == "__main__":
    analyze_chapter_lengths('raw_books/wanxiang.txt')