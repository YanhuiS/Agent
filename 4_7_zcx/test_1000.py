from fastapi import FastAPI, Request,HTTPException
from fastapi.responses import JSONResponse
import json
import networkx as nx
import os
import pandas as pd
from leader_load import read_list
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
import numpy as np
from leader_influence_read import read_influence
import random
import re
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from set import First_Name, Last_Name, Status, Trait, Interest
from collections import Counter,defaultdict
import math
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
current_item = None
network_names = []
# 允许具体的源
origins = [
    "http://localhost:8080",  # 允许的源
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 指定允许的源
    allow_credentials=True,  # 允许携带凭据
    allow_methods=["*"],     # 允许的HTTP方法
    allow_headers=["*"],     # 允许的HTTP头
)

@app.get("/NetworkInit/{json_data}")
async def NetworkInit(json_data: str):
    """
    对网络初始化，读取 follower_1w.json 文件并构建网络，构建成功返回success
    success后，调用/NetworkShow接口，展示网络信息
    """
    data  = eval(json_data)
    global network, att, likes, retweets, comments, views, colors, roles, names, genders, ages, uids, collects
    try:
        # 读取 follower_1000.json 文件并构建网络
        with open('./follower_1000.json', 'r', encoding='utf-8') as file:
            following_info = json.load(file)
        
        # 清空当前的网络图并添加节点和边
        network.clear()  # 清空当前的网络图
        network.add_nodes_from([a for a in following_info.keys()])
        network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])

        # 读取 profiles_1000.csv 文件并提取数据
        profile_path = './profiles_1000.csv'
        if os.path.exists(profile_path):
            df = pd.read_csv(profile_path, dtype={14: str})
            profiles = df.to_dict(orient='list')
        
        # 提取节点的属性值（如初始化态度）
        
        for i in range(len(profiles["id"])):
            
            att.append(profiles["init_att"][i])
            likes.append(profiles["likes"][i])
            retweets.append(profiles["retweets"][i])
            comments.append(profiles["comments"][i])
            views.append(profiles["views"][i])
            collects.append(profiles["collects"][i])
            colors.append(profiles["colors"][i])
            roles.append(profiles["roles"][i])
            names.append(profiles["names"][i])
            genders.append(profiles["genders"][i])
            ages.append(profiles["ages"][i])
            uids.append(profiles["uids"][i])

        # 构建网络数据字典
        network_data = {
            "node_att": tuple(att),  # 节点属性列表转换为元组
            "node_likes": tuple(likes),
            "node_retweets": tuple(retweets),
            "node_comments": tuple(comments),
            "node_views": tuple(views),
            "node_collects": tuple(collects),
            "node_color": tuple(colors),
            "node_role": tuple(roles),
            "node_age": tuple(ages),
            "node_gender": tuple(genders),
            "node_name": tuple(names),
            "node_uid": tuple(uids),
            "degrees": tuple(dict(network.degree()).values()),  # 每个节点的度数
            "edges": tuple(network.edges())  # 图中的边
        }

        # 打印网络数据，检查正确性
        return {"status": "success"}
    except Exception as e:
        # 捕获任何异常并返回错误信息
        print("Error initializing network:", str(e))
@app.get("/NetworkShow/{NetworkName}")
async def NetworkShow(NetworkName: str):
    """
    返回网络数据，包括节点属性、边、度数等。
    
    """
    global network, att, likes, retweets, comments, views, colors, roles, names, genders, ages, uids,collects
    try:
        # 确保 network 已被初始化
        if network is None or len(network.nodes) == 0:
            return {"status": "error", "message": "Network has not been initialized."}
        if NetworkName not in network_names:
            network_names.append(NetworkName)
        # 从 network 中提取数据
        network_data = {
            "node_att": tuple(att),  # 节点属性列表转换为元组
            "node_likes": tuple(likes),
            "node_retweets": tuple(retweets),
            "node_comments": tuple(comments),
            "node_views": tuple(views),
            "node_collects": tuple(collects),
            "node_color": tuple(colors),
            "node_role": tuple(roles),
            "node_age": tuple(ages),
            "node_gender": tuple(genders),
            "node_name": tuple(names),
            "node_uid": tuple(uids),
            "degrees": tuple(dict(network.degree()).values()),  # 每个节点的度数
            "edges": tuple(network.edges()),  # 图中的边
            "network_name": NetworkName
        }

        # 返回数据，FastAPI 会自动将其转换为 JSON 格式
        return network_data

    except Exception as e:
        return {"status": "error", "message": str(e)}
        
class Item(BaseModel):
    TF_like: bool
    TF_retweet: bool
    TF_comment: bool
    TF_view: bool
    TF_collect: bool
    TF_degree: bool
    TF_att: bool
    TF_pcomment: bool
    TF_sample:bool
    TF_leader: bool
    TF_media: bool
    cnt_dc: float
    cnt_eg: float
    cnt_t: float
    cnt_l: float
    cnt_sc: float
    cnt_sd: float
    cnt_ai: float
    cnt_ep: float
    cnt_bm: float
    cnt_c: float
    operation: str
    edges: str
@app.post("/NetworkManage/")
def NetworkManage(item: Item):
    """
    生成网络图数据
    TF_like: bool, TF_retweet: bool, TF_comment: bool, TF_view: bool, TF_collect: bool, 
    TF_degree: bool, TF_att: bool, TF_pcomment: bool（用户过滤条件，当参数为True时，即过滤网络，条件可叠加）
    TF_sample:bool, TF_leader: bool,TF_media: bool （对用户数量，意见领袖和新闻媒体进行调整）
    
    cnt_dc: float, cnt_eg: float, cnt_t: float, cnt_l: float, cnt_sc: float, cnt_sd: float, 
    cnt_ai: float, cnt_ep: float, cnt_bm: float, cnt_c: float （展示不同用户群体的占比，可以对其占比进行调整，
    通过条形图直观显示各用户群体的比例，给定十个用户类型浮点数值相加为1）
    
    operation: str, edges: str （  - 给定operation是add/delete，edges 给定 例如1_2，
    即添加user1 和user 2的连接，也可多连接，格式为1_2,3_4）
    
    """
    edge_list = item.edges.split(',')  # 将边字符串拆分成列表
    print(f"Edges to {item.operation}: {edge_list}")  # 输出边列表
    params = [item.cnt_dc, item.cnt_eg, item.cnt_t, item.cnt_l, item.cnt_sc, item.cnt_sd, item.cnt_ai, item.cnt_ep, item.cnt_bm, item.cnt_c]
    params_sum = sum(params)
    params_normalized = [p / params_sum for p in params]
    profile_path = 'profile_1w.csv'
    if os.path.exists(profile_path):
        df = pd.read_csv(profile_path)
    else:
        raise FileNotFoundError("profiles_random_data.csv 文件不存在！")
    with open('follower_1w.json', 'r', encoding='utf-8') as file:
                following_info = json.load(file)
    global network, att, likes, retweets, comments, views, colors, roles, names, genders, ages, uids, collects,current_item
    att, likes, retweets, comments, views, collects, colors, roles, names, genders, ages, uids = [], [], [], [], [], [], [], [], [], [], [], []
    if item.TF_like:
        try:
            df = df[df["likes"] >= 500]
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}

    if item.TF_retweet:
        try:
            df = df[df["retweets"] >= 50]
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    
    if item.TF_comment:
        try:
            df = df[df["comments"] >= 100]
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    if item.TF_view:
        try:
            df = df[df["views"] >= 5000]
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    if item.TF_collect:
        try:
            df = df[df["collects"] >= 50]
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    if item.TF_degree:
        try:
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            degrees = dict(network.degree())
            df["id"] = "user_" + df["id"].astype(str)
            df["degrees"] = df["id"].map(degrees)
            df = df[df["degrees"] >= 10]
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    if item.TF_att:
        try:
            df["init_att"] = df["init_att"].abs()
            df = df[df["init_att"] >= 0.1]
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    if item.TF_pcomment:
        try:
            df["init_att"] = df["init_att"].abs()
            df["init_att"] = df["init_att"]* 100
            df = df[df["init_att"] >= 50]
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    if item.TF_sample:
        try:
            df = df.sample(n=int(df.shape[0] * 0.5), random_state=42)
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    if item.TF_leader:
        try:
            leaders = df[df["roles"] == "Opinion leaders"].head(20)
            df.loc[df["roles"] == "Opinion leaders", "roles"] = 0
            df.loc[leaders.index, "roles"] = "Opinion leaders"
            others = df[df["roles"] != "Opinion leaders"]
            df = pd.concat([leaders, others])
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    if item.TF_media:
        try:
            media = df[df["roles"] == "News media"].head(20)
            df.loc[df["roles"] == "News media", "roles"] = 0
            df.loc[media.index, "roles"] = "News media"
            others = df[df["roles"] != "News media"]
            df = pd.concat([media, others])
            print(df.shape[0])
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            for i in following_info.keys():
                if int(i[5:]) not in set(df.index):
                    network.remove_node(i)
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    if sum(params_normalized) == 1:                                                    
        try:
            print(df.shape[0])
            prof_color_mapping = {
                "Arts": "red",
                "Business": "blue",
                "Communications": "green",
                "Education": "yellow",
                "Healthcare": "purple",
                "Hospitality": "orange",
                "Information Technology": "cyan",
                "Law Enforcement": "magenta",
                "Sales and Marketing": "pink",
                "Science": "brown",
                "Transportation": "grey"
            }
            prof_distribution = {
                "Arts": int(df.shape[0] * params_normalized[0]),
                "Business": int(df.shape[0] * params_normalized[1]),
                "Communications": int(df.shape[0] * params_normalized[2]),
                "Education": int(df.shape[0] * params_normalized[3]),
                "Healthcare": int(df.shape[0] * params_normalized[4]),
                "Hospitality": int(df.shape[0] * params_normalized[5]),
                "Information Technology": int(df.shape[0] * params_normalized[6]),
                "Law Enforcement": int(df.shape[0] * params_normalized[7]),
                "Sales and Marketing": int(df.shape[0] * params_normalized[8]),
                "Science": int(df.shape[0] * params_normalized[9]),
                "Transportation": int(df.shape[0] * params_normalized[10])
            }
            assigned_sum = sum(prof_distribution.values())
            if assigned_sum != df.shape[0]:
                diff = df.shape[0] - assigned_sum
                if diff > 0:
                    max_prof = max(prof_distribution, key=prof_distribution.get)
                    prof_distribution[max_prof] += diff
                elif diff < 0:
                    min_prof = min(prof_distribution, key=prof_distribution.get)
                    prof_distribution[min_prof] -= abs(diff)
            print(f"Professions distribution: {prof_distribution}")
            new_profs = []
            for prof, count in prof_distribution.items():
                new_profs.extend([prof] * count)
            random.shuffle(new_profs)
            df['profs'] = new_profs
            df['colors'] = df['profs'].map(prof_color_mapping)
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            network = nx.DiGraph()
            network.add_nodes_from([a for a in following_info.keys()])
            network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
        
    if item.operation == "add":
        try:
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            for edge in edge_list:
                user_a, user_b = f"user_{edge.split('_')[0]}", f"user_{edge.split('_')[1]}"
                if not network.has_edge(user_a, user_b):
                    network.add_edge(user_a, user_b)
                else:
                    print(f"Edge already exists: {user_a} -> {user_b}")
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
        
    if item.operation == "delete":
        try:
            att = list(df['init_att'])
            likes = list(df['likes'])
            retweets = list(df['retweets'])
            comments = list(df['comments'])
            views = list(df['views'])
            collects = list(df['collects'])
            colors = list(df['colors'])
            roles = list(df['roles'])
            names = list(df['names'])
            genders = list(df['genders'])
            ages = list(df['ages'])
            uids = list(df['uids'])
            for edge in edge_list:
                user_a, user_b = f"user_{edge.split('_')[0]}", f"user_{edge.split('_')[1]}"
                if network.has_edge(user_a, user_b):
                    network.remove_edge(user_a, user_b)
                else:
                    print("Edge does not exist.")
            network_data = {
                "node_att": tuple(att),
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 计算符合条件的节点度数
                "edges": tuple(network.edges())
            }
        except Exception as e:
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    current_item = item
    return network_data
@app.get("/NetworkParams/")
def get_network_params():
    """
    返回当前操作的Item参数
    """
    global network, att, likes, retweets, comments, views, colors, roles, names, genders, ages, uids, collects,current_item
    try:
        if current_item:
            network_data = {
                "node_att": tuple(att),  # 节点属性列表转换为元组
                "node_likes": tuple(likes),
                "node_retweets": tuple(retweets),
                "node_comments": tuple(comments),
                "node_views": tuple(views),
                "node_collects": tuple(collects),
                "node_color": tuple(colors),
                "node_role": tuple(roles),
                "node_age": tuple(ages),
                "node_gender": tuple(genders),
                "node_name": tuple(names),
                "node_uid": tuple(uids),
                "degrees": tuple(dict(network.degree()).values()),  # 每个节点的度数
                "edges": tuple(network.edges()),  # 图中的边
                "item": current_item  # 当前操作的Item
            }
            return network_data
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/NetworkList")
def get_network_list():
    """
    返回所有曾经输入过的网络名称列表
    """
    return {"status": "success", "network_names": network_names}
class Event(BaseModel):
    Describe: str
    Algorithm: str
    Time: str
    Nature: str
    Number: str
@app.post("/EventDelivery/")
async def EventDelivery(event: Event):
    """
    EventDelivery
    -事件投放，直接书写事件
    -算法文件就是json文件
    
    """
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
    """
    AgentManage
    -agent管理，直接书写agent
    -算法文件就是json文件
    """
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
    """
    EventDelivery
    -visibility：可见性
    -content：内容
    -time：时间
    -att：态度
    """
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