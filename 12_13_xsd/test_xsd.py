import argparse
from http.client import HTTPException
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
from datetime import datetime, timedelta
import random
import pandas as pd
import os
import re

app = FastAPI()
@app.get("/IntervensionImpact/{event_id}/{algorithm_id}")
async def UserAttitudeDynamics(event_id: str,algorithm_id: str):
    """
    功能:统计在某个节点一引入干预算法后,用户支持中立反对态度随时间变化的动态数据
    输入参数:event_id: 事件id, algorithm_id: 干预算法id
    返回值:一个列表,列表中每个元素是一个字典。time:2022-01-01,support:100,neutral:80,oppose:20;
    """
    # 读取attitude_impacted_data_{event_id}_{algorithm_id}.csv
    result = pd.read_csv(f'/root/autodl-tmp/syh/12_13_xsd/attitude_impacted_data_{event_id}_{algorithm_id}.csv')

    output = tuple(result.to_dict(orient='records'))

    return output

@app.get("/UserAttitudeChange/{event_id}/{algorithm_id}")
async def UserAttitudeChange(event_id: str,algorithm_id: str):
    """
    功能:展示在某个节点一引入干预算法后,粉丝数最多的前50名用户的评论态度变化,提取出情感词,并分析情感词的情感倾向
    输入参数:event_id: 事件id, algorithm_id: 干预算法id,
    返回值:一个列表,列表中每个元素是一个字典。userinfo:{userID,name,fans_num,gender,figure},pre_content:{time,content,attitude,emotion_words:{word1{word_attitude:positive, word_content: happy},word2,...}},post_content:{time,content,attitude,emotion_words:{word1{word_attitude:negative, word_content: sad},word2,...}};
    """
    # 读取爬取的用户评论数据
    mate_data = pd.read_csv(f'/root/autodl-tmp/syh/12_13_xsd/user_comment_data_{event_id}_{algorithm_id}.csv')
    
    # 结合用户数据,筛选粉丝量最多的50人
    top50_user = mate_data.groupby('userID').size().sort_values(ascending=False).head(50).index.tolist()
    mate_data = mate_data[mate_data['userID'].isin(top50_user)]

    # 结合干预算法id，选取最近一次干预前后的两个评论，作为pre_content和post_content(包含time,content,attitude)
    mate_data = mate_data.sort_values(by=['userID','time'])
    result = mate_data.groupby('userID').apply(get_pre_post_content).reset_index(drop=True)
    
    # 使用paddlenlp 提取情感词并分析词的倾向
    result['pre_content']['emotion_words'] = result['pre_content']['content'].apply(lambda x: get_emotion_words(x))
    result['post_content']['emotion_words'] = result['post_content']['content'].apply(lambda x: get_emotion_words(x))
    
    return result

def get_emotion_words(content):
    """
    使用paddlenlp 提取情感词并分析词的倾向
    输入参数:content: 评论内容
    """
    return

def get_pre_post_content(df):
    """
    结合干预算法id,选取最近一次干预前后的两个评论,作为pre_content和post_content(包含time,content,attitude)
    输入参数:df: 用户评论数据
    返回值:pre_content,post_content
    """
    
    # 选取和干预时间intervention_time 最接近的两个评论
    pre_content = df[df['time'] < df['intervention_time']].iloc[-1]
    post_content = df[df['time'] > df['intervention_time']].iloc[0]

    return pre_content,post_content

def get_top50_user(df):
    """
    选取评论数最多的50个用户
    输入参数:df: 用户评论数据
    """
    return df.groupby('userID').size().sort_values(ascending=False).head(50).index.tolist()

# 和UserAttitudeChange基本一样 
@app.get("/GroupAttitudeChange/")
async def GroupAttitudeChange(event_id: int, algorithm_id: int):
    """
    功能:统计在某个节点一引入干预算法后,群体支持中立反对态度随时间变化的动态数据
    输入参数:event_id: 事件id, algorithm_id: 干预算法id
    返回值:一个列表,列表中每个元素是一个字典。time:2022-01-01,support:100,neutral:80,oppose:20;
    """
    # 读取attitude_impacted_data_{event_id}_{algorithm_id}.csv
    result = pd.read_csv(f'/root/autodl-tmp/syh/12_13_xsd/attitude_impacted_data_{event_id}_{algorithm_id}.csv')

    output = tuple(result.to_dict(orient='records'))

    return output

@app.get("/GeographicalAttitudes/{event_id}/{algorithm_id}}/{country}/{time}")
async def GeographicalAttitudes(event_id: str, algorithm_id: str, country: str, time: str):
    """
    功能：根据输入的事件描述、国家和时间，返回该国家各省态度数据
    输入参数：eventdesc: 事件描述，country: 国家，time: 时间
    返回值：一个列表，列表中每个元素是一个字典。
    [
  {
    "USA": "New York",
    "USA_attitude": 0.5
  },
  {
    "USA": "Los Angeles",
    "USA_attitude": 0.3
  }
    ]
    """
    geo_data = pd.read_csv(f'/root/autodl-tmp/syh/12_13_xsd/geo_data_{event_id}_{algorithm_id}.csv')
    # 根据 country 参数查找对应的列
    if country == "China":
        country_column = "China"
        attitude_column = "China_attitude"
    elif country == "USA":
        country_column = "USA"
        attitude_column = "USA_attitude"
    else:
        raise HTTPException(status_code=404, detail="Country not found in the data")
    
    # 检查列是否存在
    if country_column not in geo_data.columns or attitude_column not in geo_data.columns:
        raise HTTPException(status_code=404, detail="Required columns not found in the data")
    
    # 构建结果字典元组
    result_list = []
    for i in range(len(geo_data)):
        province = geo_data[country_column][i]
        attitude_value = geo_data[attitude_column][i]
        result_list.append({country_column: province, attitude_column: attitude_value})
    
    return tuple(result_list)

@app.get("/GeographicalAttitudesChange/{event_id}/{algorithm_id}/{country}")
async def get_geographical_attitudes_change(event_id: int, algorithm_id: int, country: str):
    """
    功能：根据输入的事件描述、国家和时间，返回该国家各省干预前后态度数据之差(以干预前为基准+-)
    输入参数：eventdesc: 事件描述，country: 国家，time: 时间
    返回值：一个列表，列表中每个元素是一个字典。
    [
  {
    "USA": "New York",
    "USA_attitude": +0.1
  },
  {
    "USA": "Los Angeles",
    "USA_attitude": -0.3
  }
    ]
    """

    geo_data = pd.read_csv(f'/root/autodl-tmp/syh/12_13_xsd/geo_data_{event_id}_{algorithm_id}.csv')
    # 根据 country 参数查找对应的列
    if country == "China":
        country_column = "China"
        attitude_column = "China_attitude"
    elif country == "USA":
        country_column = "USA"
        attitude_column = "USA_attitude"
    else:
        raise HTTPException(status_code=404, detail="Country not found in the data")
    
    # 检查列是否存在
    if country_column not in geo_data.columns or attitude_column not in geo_data.columns:
        raise HTTPException(status_code=404, detail="Required columns not found in the data")
    
    
    