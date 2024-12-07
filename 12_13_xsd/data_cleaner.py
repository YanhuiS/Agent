import argparse
import pandas as pd

def save_content_to_text(file_path, column_name, output_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    
    # 检查列是否存在
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")
    
    # 提取指定列内容，并替换换行符
    content = df[column_name].apply(lambda x: x.replace('\n', ' ')).tolist()
    content = df[column_name].apply(lambda x: x.replace('\r', ' ')).tolist()
    content = df[column_name].apply(lambda x: x.replace('\r\n', ' ')).tolist()
    content = df[column_name].apply(lambda x: x.replace('\t', ' ')).tolist()
    
    # 将内容保存为文本文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in content:
            f.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract content from a CSV file and save it as a text file.")
    parser.add_argument("--file_path", type=str, required=True, help="The path to the CSV file.")
    parser.add_argument("--column_name", type=str, default="Content", help="The name of the column to extract content from.")
    parser.add_argument("--output_path", type=str, required=True, help="The path to save the output text file.")
    
    args = parser.parse_args()
    
    save_content_to_text(args.file_path, args.column_name, args.output_path)
