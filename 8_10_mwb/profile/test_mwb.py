from fastapi import FastAPI, Request,HTTPException, Query
from fastapi.responses import JSONResponse
import json
import networkx as nx
import os
import pandas as pd
from leader_load import read_list
from datetime import datetime, timedelta
from collections import Counter,defaultdict
from fastapi.responses import JSONResponse
import numpy as np
from leader_influence_read import read_influence
import random
import re
from set import First_Name, Last_Name, Status, Trait, Interest

app = FastAPI()

@app.get("/UserAttributeStatistics/{eventdesc}")
async def UserAttributeStatistics(eventdesc: str):
    """
    功能：统计不同用户属性下（包括性别，年龄，人种，受教育程度，社会角色，经济水平，社交能力）的群体对事件的支持中立反对意见分布
    输入参数：eventdesc: 事件描述
    返回值：一个列表，列表中每个元素是一个字典，字典的键是用户属性，值是对应的支持中立反对意见分布
    """
    try:
        # 读取CSV文件
        result = pd.read_csv('./8_10_mwb/profile/1000profile.csv')
        
        # 初始化结果字典
        statistics = {}

        # 统计性别
        for gender in ['male', 'female']:
            positive_count = len(result[(result['gender'] == gender) & (result['init_att'] > 0)])
            negative_count = len(result[(result['gender'] == gender) & (result['init_att'] <= 0)])
            statistics[gender] = {'positive': positive_count, 'negative': negative_count}

        # 统计年龄
        age_groups = {
            '18-29': (18, 29),
            '30-49': (30, 49),
            '50-64': (50, 64),
            '65+': (65, 100)
        }
        for group, (min_age, max_age) in age_groups.items():
            positive_count = len(result[(result['age'] >= min_age) & (result['age'] <= max_age) & (result['init_att'] > 0)])
            negative_count = len(result[(result['age'] >= min_age) & (result['age'] <= max_age) & (result['init_att'] <= 0)])
            statistics[group] = {'positive': positive_count, 'negative': negative_count}

        # 统计人种
        for race in result['racist'].unique():
            positive_count = len(result[(result['racist'] == race) & (result['init_att'] > 0)])
            negative_count = len(result[(result['racist'] == race) & (result['init_att'] <= 0)])
            statistics[race] = {'positive': positive_count, 'negative': negative_count}

        # 统计教育程度
        for education in result['education'].unique():
            positive_count = len(result[(result['education'] == education) & (result['init_att'] > 0)])
            negative_count = len(result[(result['education'] == education) & (result['init_att'] <= 0)])
            statistics[education] = {'positive': positive_count, 'negative': negative_count}

        # 统计经济水平
        for economic in result['economic'].unique():
            positive_count = len(result[(result['economic'] == economic) & (result['init_att'] > 0)])
            negative_count = len(result[(result['economic'] == economic) & (result['init_att'] <= 0)])
            statistics[economic] = {'positive': positive_count, 'negative': negative_count}

        # 统计社会角色
        for status, categories in  Status.items():
            positive_count = len(result[(result['status'].isin(categories)) & (result['init_att'] > 0)])
            negative_count = len(result[(result['status'].isin(categories)) & (result['init_att'] <= 0)])
            statistics[status] = {'positive': positive_count, 'negative': negative_count}

        return statistics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")


@app.get("/OpinionLeaderInfluence/{eventdesc}")
async def OpinionLeaderInfluence(eventdesc: str):
    """
    功能：统计事件中的意见领袖对事件的影响力，伴随时间变化
    输入参数：eventdesc: 事件描述
    返回值：一个列表，列表中每个元素是一个字典。name：bob，time1：350；name：bob，time2：300；
    """
    try:
        folder_path = './9_11_wwt/leader/'
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # 初始化一个字典，用于存储每个日期的用户影响力
        influence_data = defaultdict(lambda: defaultdict(int))
        
        # 遍历文件夹内的所有CSV文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                user = os.path.splitext(filename)[0]  # 从文件名中获取用户名
                try:
                    df = pd.read_csv(file_path)
                    
                    # 检查是否有createdAt和likeCount列
                    if 'createdAt' not in df.columns or 'likeCount' not in df.columns:
                        raise HTTPException(status_code=400, detail=f"File {file_path} does not contain required columns")
                    
                    # 计算每个用户在每个日期的影响力
                    df['createdAt'] = pd.to_datetime(df['createdAt']).dt.date
                    grouped = df.groupby('createdAt')['likeCount'].sum().reset_index()
                    
                    for _, row in grouped.iterrows():
                        date = row['createdAt']
                        like_count = row['likeCount']
                        influence_data[date][user] += like_count
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error processing file {file_path}: {e}")
        
        # 构建结果列表
        result = []
        for date, users in influence_data.items():
            sorted_users = sorted(users.items(), key=lambda x: x[1], reverse=True)
            for user, like_count in sorted_users:
                result.append({"date": str(date), "user": user, "influence": like_count})
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")

@app.get("/WordCloudAnalysis/{eventdesc}/{time_interval}")
async def WordCloudAnalysis(eventdesc: str, time_interval: str):
    """
    功能：统计与事件相关的关键词及词频
    输入参数：eventdesc: 事件描述，time_interval: 时间区间
    返回值：一个列表，列表中每个元素是一个字典。word：中国，frequency：100；word：美国，frequency：80；
    """
    try:
        # 解析时间区间
        start_date_str, end_date_str = time_interval.split(',')
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # 初始化词频计数器
        word_counts = Counter()
        
        # 遍历时间区间内的每一天
        current_date = start_date
        while current_date <= end_date:
            file_path = f'./8_10_mwb/2024/{current_date.strftime("%Y-%m-%d")}.json'
            if os.path.exists(file_path):
                # 读取JSON文件
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data_list = json.load(file)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Error decoding JSON in file {file_path}: {e}")
                
                for data in data_list:
                    # 检查是否有content键
                    if 'content' not in data:
                        raise HTTPException(status_code=400, detail=f"File {file_path} does not contain 'content' key")
                    
                    # 统计词频
                    content = data['content']
                    words = re.findall(r'[\u4e00-\u9fff]+|\w+', content.lower())
                    word_counts.update(words)
            
            current_date += timedelta(days=1)
        
        # 获取词频前一百的词语
        top_words = word_counts.most_common(100)
        
        # 转换为字典列表
        top_words_list = [{"word": word, "frequency": count} for word, count in top_words]
        
        return top_words_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {e}")



@app.get("/UserAttitudeDynamics/{eventdesc}/{time_scale}")
async def UserAttitudeDynamics(eventdesc: str, time_scale: str):
    """
    功能：统计用户支持中立反对态度随时间变化的动态数据
    输入参数：eventdesc: 事件描述，time_scale: 时间尺度
    返回值：一个列表，列表中每个元素是一个字典。time：2022-01-01，support：100，neutral：80，oppose：20；
    """
    try:
        folder_path = './8_10_mwb/2024'
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # 初始化结果字典
        attitude_dynamics = {}

        # 遍历文件夹内的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data_list = json.load(file)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Error decoding JSON in file {file_path}: {e}")
                
                support_count = 0
                neutral_count = 0
                oppose_count = 0
                
                for data in data_list:
                    # 检查是否有stance键
                    if 'stance' not in data:
                        raise HTTPException(status_code=400, detail=f"File {file_path} does not contain 'stance' key")
                    
                    # 统计stance列的值
                    stance = data['stance']
                    if stance == 'support':
                        support_count += 1
                    elif stance == 'neutral':
                        neutral_count += 1
                    elif stance == 'oppose':
                        oppose_count += 1
                
                # 获取日期
                date = filename.split('.')[0]
                
                # 存储结果
                attitude_dynamics[date] = {
                    'support': support_count,
                    'neutral': neutral_count,
                    'oppose': oppose_count
                }
        
        return attitude_dynamics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")

    
@app.get("/RealtimeData/{eventdesc}/{time}")
async def RealtimeData(eventdesc: str, time: str):
    """
    功能：根据输入的时间，返回该时间点的实时数据
    输入参数：eventdesc: 事件描述，time: 时间
    返回值：字典，
    {
        "时间": "2024-11-23",
        "用户参与度": 0.85,
        "态度占比": {"support": 0.5, "neutral": 0.3, "oppose": 0.2},
        "话题热度": "hot",
        "用户留存率": 0.75,
        "每日新增用户": 100,
        "信息茧房效应": 0.45
    }
    """
    try:
        # 将输入的时间字符串转换为datetime格式，只具体到天
        input_time = datetime.strptime(time, "%Y-%m-%d")
        date_str = input_time.strftime("%Y-%m-%d")
        
        # 读取时间名为该时间的文件
        file_path = f'./8_10_mwb/2024/{date_str}.json'
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)
        
        # 计算用户参与度
        user_engagement = len(data_list) / 10000
        
        # 计算态度占比
        stance_counts = {"support": 0, "neutral": 0, "oppose": 0}
        for entry in data_list:
            stance = entry.get("stance")
            if stance in stance_counts:
                stance_counts[stance] += 1
        total_stances = sum(stance_counts.values())
        attitude_ratio = {k: v / total_stances for k, v in stance_counts.items()}
        
        # 计算话题热度
        total_comments = sum(entry.get("comments_count", 0) for entry in data_list)
        topic_heat = "hot" if len(data_list) + total_comments > 100 else "cool"
        
        # 计算用户留存率和每日新增用户
        previous_date_str = (input_time - timedelta(days=1)).strftime("%Y-%m-%d")
        previous_file_path = f'./8_10_mwb/2024/{previous_date_str}.json'
        if os.path.exists(previous_file_path):
            with open(previous_file_path, 'r', encoding='utf-8') as file:
                previous_data = json.load(file)
            daily_new_users = len(previous_data) - len(data_list)
            previous_ids = {entry["_id"] for entry in previous_data}
            current_ids = {entry["_id"] for entry in data_list}
            retained_users = len(previous_ids & current_ids)
            user_retention_rate = retained_users / len(previous_ids) if previous_ids else 1
        else:
            user_retention_rate = 1
            daily_new_users = len(data_list)
        
        # 生成信息茧房效应的随机值
        filter_bubble_effect = random.uniform(0, 1)
        
        # 构建结果字典
        result_dict = {
            "时间": time,
            "用户参与度": user_engagement,
            "态度占比": attitude_ratio,
            "话题热度": topic_heat,
            "用户留存率": user_retention_rate,
            "每日新增用户": daily_new_users,
            "信息茧房效应": filter_bubble_effect
        }
        
        return result_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")



@app.get("/GeographicalAttitudes/{eventdesc}/{country}")
async def GeographicalAttitudes(eventdesc: str, country: str):
    """
    功能：根据输入的事件描述、国家和时间，返回该国家各省态度数据
    输入参数：eventdesc: 事件描述，country: 国家，time: 时间
    返回值：一个列表，列表中每个元素是一个字典。
    """
    try:
        folder_path = './8_10_mwb/2024'
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # 初始化结果字典
        location_attitude = {}
        location_counts = {}

        # 遍历文件夹内的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data_list = json.load(file)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Error decoding JSON in file {file_path}: {e}")
                
                for data in data_list:
                    # 检查是否有ip_location和att键
                    if 'ip_location' not in data or 'att' not in data:
                        raise HTTPException(status_code=400, detail=f"File {file_path} does not contain required keys")
                    
                    # 统计ip_location每个元素的att列的态度分数
                    ip_location = data['ip_location']
                    att = data['att']
                    
                    if ip_location not in location_attitude:
                        location_attitude[ip_location] = 0
                        location_counts[ip_location] = 0
                    
                    location_attitude[ip_location] += att
                    location_counts[ip_location] += 1
        
        # 计算平均值
        for location in location_attitude:
            location_attitude[location] /= location_counts[location]
        
        return location_attitude
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")




@app.get("/ForwardingTrends/{eventdesc}")
async def ForwardingTrends(eventdesc: str):
    """
    功能：统计事件不同时间的转发量
    输入：事件描述（eventdesc）
    返回值：一个列表，列表中每个元素是一个字典。time：2022-01-01 00:00:00，total：100；
    """
    folder_path = './8_10_mwb/2024'
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    
    # 初始化结果字典
    forwarding_trends = {}

    # 遍历文件夹内的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data_list = json.load(file)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Error decoding JSON in file {file_path}: {e}")
            
            for data in data_list:
                # 检查是否有created_at和comments_count键
                if 'created_at' not in data or 'comments_count' not in data:
                    continue
                
                # 获取时间和评论数
                created_at = datetime.strptime(data['created_at'], "%Y-%m-%d %H:%M:%S")
                comments_count = data['comments_count']
                
                # 计算时间段起点
                time_slot = created_at.replace(minute=0, second=0, microsecond=0)
                if created_at.hour % 2 != 0:
                    time_slot -= timedelta(hours=1)
                
                time_slot_str = time_slot.strftime("%Y-%m-%d %H:%M:%S")
                
                if time_slot_str not in forwarding_trends:
                    forwarding_trends[time_slot_str] = 0
                
                forwarding_trends[time_slot_str] += 1 + comments_count
    
    # 转换结果为列表
    output = [{"time": time, "total": total} for time, total in forwarding_trends.items()]
    
    return output




@app.get("/KeyUsers/{eventdesc}/{ID}")
async def KeyUsers(eventdesc: str,ID: str):
    """
    功能：根据输入的事件描述和ID，返回该ID的姓名，头像，粉丝，转发内容等信息
    输入：事件描述（eventdesc），ID（str）
    返回值：一个字典，包含姓名，头像，粉丝，转发内容等信息
    """
    folder_path = './8_10_mwb/2024'
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    
    ID = ID.strip()
    
    # 遍历文件夹内的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data_list = json.load(file)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Error decoding JSON in file {file_path}: {e}")
            
            for data in data_list:
                # 检查是否有_id键
                if '_id' not in data:
                    continue
                
                # 查找第一个输入ID与字典_id键的值一样的字典
                if data['_id'] == ID:
                    result_dict = {
                        "created_at": data.get("created_at"),
                        "content": data.get("content"),
                        "reposts_count": data.get("reposts_count"),
                        "comments_count": data.get("comments_count"),
                        "attitudes_count": data.get("attitudes_count"),
                        "user.nick_name": data.get("user", {}).get("nick_name"),
                        "user.avatar_hd": data.get("user", {}).get("avatar_hd")
                    }
                    return result_dict
    
    raise HTTPException(status_code=404, detail="ID not found in the data")



@app.get("/CriticalUserPaths/{eventdesc}")
async def CriticalUserPaths(eventdesc: str):
    """
    功能：根据输入的事件描述，返回该事件的相关关键用户传播路径
    输入：事件描述（eventdesc）
    返回值：一个字典，有nodes和links两个键，分别对应节点和边的信息；节点信息包括id，name，avatar等，边信息包括源节点和目标节点
    """
    folder_path = './8_10_mwb/2024'
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    
    posts = []

    # 遍历文件夹内的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data_list = json.load(file)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Error decoding JSON in file {file_path}: {e}")
            
            for data in data_list:
                # 计算总互动数
                total_interactions = data.get('reposts_count', 0) + data.get('comments_count', 0) + data.get('attitudes_count', 0)
                data['total_interactions'] = total_interactions
                posts.append(data)
    
    # 找出总互动数最大的十个帖子
    top_posts = sorted(posts, key=lambda x: x['total_interactions'], reverse=True)[:10]
    
    # 构建网络
    G = nx.DiGraph()
    
    # 添加节点
    for post in top_posts:
        G.add_node(post['_id'], **post)
    
    # 添加边
    for i, post in enumerate(top_posts):
        created_at = datetime.strptime(post['created_at'], "%Y-%m-%d %H:%M:%S")
        for j in range(i + 1, len(top_posts)):
            next_post = top_posts[j]
            next_created_at = datetime.strptime(next_post['created_at'], "%Y-%m-%d %H:%M:%S")
            if next_created_at - created_at >= timedelta(hours=12):
                G.add_edge(post['_id'], next_post['_id'])
    
    # 构建结果字典
    result = {
        "nodes": [{**G.nodes[node], "id": node} for node in G.nodes],
        "links": [{"source": source, "target": target} for source, target in G.edges]
    }
    
    return result

