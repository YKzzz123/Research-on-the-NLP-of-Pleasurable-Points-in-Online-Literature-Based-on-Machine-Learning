# -*- coding: utf-8 -*-
"""截取 [book_name]_all.txt 章节内容保存到 [book_name].txt"""

def extract_chapters():
    input_file = 'raw_books/row_text/wanxiang_all.txt'
    output_file = 'raw_books/wanxiang.txt'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 此处修改要分割的章节
    start_marker = '第九百八十八章' # 起始章开始
    next_chapter_marker = '第一千零六十四章'  # 终止章结束的下一行
    
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if start_marker in line:
            start_line = i
            break
    
    if start_line is None:
        raise ValueError(f'未找到章节起始标记: {start_marker}')
    
    for i in range(start_line + 1, len(lines)):
        if next_chapter_marker in lines[i]:
            end_line = i - 1  # 起始章节最后一行的索引
            break
    
    if end_line is None:
        raise ValueError(f'未找到章节结束位置 (起始章之后应为{next_chapter_marker})')
    
    content = ''.join(lines[start_line:end_line + 1])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f'起始行: {start_line + 1} ({lines[start_line].strip()})')
    print(f'结束行: {end_line + 1} ({lines[end_line].strip()})')
    print(f'共 {end_line - start_line + 1} 行，已保存到 {output_file}')

if __name__ == '__main__':
    extract_chapters()
