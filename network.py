from fastapi import FastAPI
import json

import networkx as nx
import os
import pandas as pd
from leader_load import read_list
from datetime import datetime
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}



@app.get("/NetworkForEvent/{eventdesc}")
async def read_network_for_event(eventdesc: str):
    with open('./follower_1000.json', 'r', encoding='utf-8') as file:
        following_info = json.load(file)
    network = nx.DiGraph()
    network.add_nodes_from([a for a in following_info.keys()])
    network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])

    # print("Edges", network.edges())
    profile_path = './profiles_1000.csv'
    if os.path.exists(profile_path):
        df = pd.read_csv(profile_path)
        profiles = df.to_dict(orient='list')

    val = []
    for i in range(len(profiles["id"])):
        val.append(profiles["init_att"][i])
    # print("val", val)network
    network_dt1 = {"node_val": tuple(val), "degrees":tuple(dict(network.degree()).values()), "edges": tuple(network.edges())}
    # print(network_dt1)

    networks_dt = {"2024-11-16 21:36:18": network_dt1, "2024-11-17 21:36:18": network_dt1, "2024-11-18 21:36:18": network_dt1, "2024-11-19 21:36:18": network_dt1} 

    return networks_dt


