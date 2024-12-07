from fastapi import FastAPI
import json
import networkx as nx
import os
import pandas as pd
from leader_load import read_list
from datetime import datetime
from fastapi.responses import JSONResponse
import numpy as np
from leader_influence_read import read_influence
import random
app = FastAPI()
network = nx.DiGraph()
# network = nx.DiGraph()
# network.remove_nodes_from(network.nodes)
att = []
likes = []
retweets = []
comments = []
views = []
collects = []
colors = []
roles = []
genders = []
ages = []
names = []
uids = []
@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI"}
class Event(BaseModel):
    Describe: str
    Algorithm: str
    Time: str
    Nature: str
    Number: str
@app.post("/EventDelivery/")
async def EventDelivery(event: Event):
    try:
        Describe = event.Describe
        Algorithm  = event.Algorithm
        Time = event.Time
        Nature = event.Nature
        Number = event.Number
        demo_data = {
            "Describe": Describe,
            "Algorithm": Algorithm,
            "Time": Time,
            "Nature": Nature,
            "Number": Number
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return demo_data
class Agent(BaseModel):
    opertion: bool
    config: str
    number: str
    att: str
    char: str
    intense: str
    breadth: str
@app.post("/AgentManage/")
async def AgentManage(agent: Agent):
    try:
        if agent.opertion == True:
            demo_data = agent.config
        else:
            demo_data = {
                "number": agent.number,
                "att": agent.att,
                "char": agent.char,
                "intense": agent.intense,
                "breadth": agent.breadth
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return demo_data

@app.get("/Topic/{visibility}/{content}/{time}/{att}")
async def EventDelivery(visibility:str, content:str, time:str, att:str):
    try:
        visibility = "男性"
        content = "意见分享"
        time = "2024-10-26"
        att = "积极"
        demo_data = {
            "visibility": visibility,
            "content": content,
            "time": time,
            "att": att
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return demo_data

@app.get("/Agent_list/{event_id}")
async def agent_list( event_id: str):
    """
    获取agent列表

    - event_id: event的id
    - 返回agent列表，包含用户头像、昵称、ID、话题内容、粉丝数、点赞量、转发量
    """
    df = pd.read_excel('/root/autodl-tmp/syh/test_influence.xlsx')
    
    df = df.rename(columns={
        'name': 'user_name',
        'uid': 'user_id',
        'content': 'topic_content',
        'replycount': 'replycount',
        'retweetcount': 'retweetcount',
        'favoritecount': 'favoritecount',
    })
    
    
    # 选择需要的列
    result_df = df[['user_name', 'user_id', 'topic_content', 'replycount', 'retweetcount', 'favoritecount','avatar']]
    
    # 将DataFrame转换为字典列表
    result_list = result_df.to_dict('records')
    
    return tuple(result_list)