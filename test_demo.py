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

att, likes, retweets, comments, views, collects, colors, roles, names, genders, ages, uids = [], [], [], [], [], [], [], [], [], [], [], []
profile_path = './profiles_random_data.csv'
if os.path.exists(profile_path):
    df = pd.read_csv(profile_path)
else:
    raise FileNotFoundError("profiles_random_data.csv 文件不存在！")
random_sample = df.sample(n=500, random_state=42)
att = list(random_sample['init_att'])
with open('./follower_1000.json', 'r', encoding='utf-8') as file:
    following_info = json.load(file)
network = nx.DiGraph()
network.add_nodes_from([a for a in following_info.keys()])
network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])
for i in following_info.keys():
    if int(i[5:]) not in set(random_sample.index):
        cnt +=1
        network.remove_node(i)

# print(network.nodes)
# print(network.edges)
# print(cnt)
    
