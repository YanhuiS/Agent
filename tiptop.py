from fastapi import FastAPI
import json

import networkx as nx
import os
import pandas as pd
from leader_load import read_list
from datetime import datetime
from fastapi.responses import JSONResponse
import numpy as np



def tip_point(time):
    try:
        # 将输入的时间字符串转换为datetime对象
        time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        print(f"Time limit: {time_limit}")
            
        # 使用pandas读取Excel文件
        df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_influ/17980523.csv')
        print(f"DataFrame loaded with {len(df)} rows")
            
        # 确保'createdb'列是datetime类型
        df['createdb'] = pd.to_datetime(df['createdb'])
            
        # 筛选出'createdb'小于time_limit的所有行
        df_filtered = df[df['createdb'] < time_limit]
        print(f"Filtered rows: {len(df_filtered)}")
            
        # 打印'retweetcount'列的数据类型和非空值的数量
        print(f"Retweet count data type: {df_filtered['retweetcount'].dtype}")
        print(f"Retweet count non-null count: {df_filtered['retweetcount'].notnull().sum()}")
            
        # 处理NaN和无穷大的值
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=['retweetcount'])
            
        # 按照'retweetcount'列从大到小排序
        df_sorted = df_filtered.sort_values(by='retweetcount', ascending=False)
            
        # 选取前十个结果
        df_top10 = df_sorted.head(10)
        print(f"Top 10 rows: {len(df_top10)}")
            
        # 将结果转换为字典列表，以便JSONResponse可以序列化
        result_dict = df_top10.to_dict(orient='records')
            
        result1=tuple(result_dict)
        return result1
        
    except Exception as e:
        return {"error": str(e)}



# 使用函数的例子
time_input = "2024-11-15 10:00:00"
result = tip_point(time_input)
print(result)