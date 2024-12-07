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


@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}



@app.get("/NetworkForEvent/{eventid}/{dt}")
async def read_network_for_event(eventid: str, dt: str):
    """
    功能：根据事件id和日期，返回该事件在该日期的社交网络图，不同时间用户的态度值不同，用户的节点大小颜色也不同。
    输入参数：eventid：事件id，dt：日期
    返回值：该时间的社交网络路
    """
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

    # networks_dt = {"2024-11-16 21:36:18": network_dt1, "2024-11-17 21:36:18": network_dt1, "2024-11-18 21:36:18": network_dt1, "2024-11-19 21:36:18": network_dt1} 
    networks_dt = {dt: network_dt1} 
    return networks_dt



@app.get("/LeaderChoose/{eventdesc}")
async def leader_choose(eventdesc: str):
    # 尝试读取Excel文件
    try:
        # with open('/root/autodl-tmp/syh/9_11_wwt/leader_info/leader_info.csv', 'rb') as f:
        #     result = chardet.detect(f.read())
        #     encoding = result['encoding']
        # # 使用pandas读取Excel文件
        # df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_info/leader_info.csv',  encoding=encoding)
        df = pd.read_excel('/root/autodl-tmp/syh/9_11_wwt/leader_info/leader_info1.xlsx')
        
        # 检查所需的列是否存在
        required_columns = ['name', 'uid','favourites_count', 'followers_count', 'listed_count', 'content', 'replycount', 'retweetcount', 'favoritecount','createdb','avatar']
        for column in required_columns:
            if column not in df.columns:
                print(f"The column '{column}' is missing in the file.")
                return None
        
        # 提取所需的列
        result = df[required_columns]
    except FileNotFoundError:
        print("The file was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    column_mapping = {
    'name': 'leader',
    'uid': 'leader_id',
    'favourites_count': 'leader_fan_count',
    'followers_count': 'leader_follower_count',
    'listed_count': 'leader_post_count',
    'favoritecount': 'leader_liked_count',
    'retweetcount': 'leader_repost_count',
    'replycount': 'leader_comment_count',
    'avatar': 'leader_pic',
    'createdb': 'create_time',
    }
    result = result.rename(columns=column_mapping)


    result1=result[['leader','leader_id']]
    leader_info = tuple(result1.to_dict(orient='records'))
    return leader_info

@app.get("/LeaderBasicInfo/{leader_id}/{event_id}")
async def leader_basic_info(leader_id:str, event_id: str):
    """
    获取leader基本信息

    - leader_id: leader的id
    - event_id: event的id
    - 返回意见领袖姓名，性别，粉丝数、关注数、作品数、点赞量、转发量、评论量、头像。
    """
    try:
        # with open('/root/autodl-tmp/syh/9_11_wwt/leader_info/leader_info.csv', 'rb') as f:
        #     result = chardet.detect(f.read())
        #     encoding = result['encoding']
        # # 使用pandas读取Excel文件
        # df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_info/leader_info.csv',  encoding=encoding)
        df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_info/leader_info1.csv')
        
        # 检查所需的列是否存在
        required_columns = ['name', 'uid','favourites_count', 'followers_count', 'listed_count', 'content', 'replycount', 'retweetcount', 'favoritecount','createdb','avatar']
        for column in required_columns:
            if column not in df.columns:
                print(f"The column '{column}' is missing in the file.")
                return None
        
        # 提取所需的列
        result = df[required_columns]
    except FileNotFoundError:
        print("The file was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    column_mapping = {
    'name': 'leader',
    'uid': 'leader_id',
    'favourites_count': 'leader_fan_count',
    'followers_count': 'leader_follower_count',
    'listed_count': 'leader_post_count',
    'favoritecount': 'leader_liked_count',
    'retweetcount': 'leader_repost_count',
    'replycount': 'leader_comment_count',
    'avatar': 'leader_pic',
    'createdb': 'create_time',
    }
    result = result.rename(columns=column_mapping)


    specific_id = int(leader_id)
    specific_row = result.loc[result['leader_id'] == specific_id]

    # # 使用query方法根据id查找特定的行
    # specific_id = id
    # specific_row = result.query('leader_id == @specific_id')

    required_columns = ['leader', 'leader_id','leader_fan_count', 'leader_follower_count',  'leader_post_count','leader_liked_count', 'leader_repost_count', 'leader_comment_count', 'leader_pic']
    # leader_info1 = specific_row[required_columns]
    # leader_info_select = tuple(leader_info1.to_dict(orient='records'))
    leader_info_select = tuple(specific_row[required_columns].to_dict(orient='records'))

    return leader_info_select

@app.get("/TipPoint/{time}/{event_id}")
async def tip_point(time:str, event_id: str):
    """
    获取引爆点用户数据

    - time: 时间限制(2022-10-05 10:00:00)
    - event_id: event的id
    - 返回引爆点用户姓名、id、粉丝数、转发数、创作时间、头像。
    """
    try:
        # 将输入的时间字符串转换为datetime对象
        time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        print(f"Time limit: {time_limit}")
        
        # 使用pandas读取Excel文件
        df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/comment/comment.csv')
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

        
        # **在这里先过滤掉 uid 为 NaN 的行**
        df_filtered = df_filtered.dropna(subset=['uid'])
        
        # 按照'retweetcount'列从大到小排序
        df_sorted = df_filtered.sort_values(by='retweetcount', ascending=False)
        print(df_sorted)

        df_sorted_unique = df_sorted.drop_duplicates(subset='name', keep='first')
        # 选取前十个结果
        # df_top10 = df_sorted.head(10)
        df_top10 = df_sorted_unique.iloc[:10]
        print(df_top10)
        print(f"Top 10 rows: {len(df_top10)}")

        df_top10['createdb'] = df_top10['createdb'].dt.strftime('%Y-%m-%d %H:%M:%S')  # 转换为字符串
        
        # 将结果转换为字典列表，以便JSONResponse可以序列化
        required_columns = ['name', 'uid','favoritecount', 'retweetcount', 'createdb','avatar','content']
        leader_info_select = tuple(df_top10[required_columns].to_dict(orient='records'))

        def clean_data(data):
            return [
                {key: (None if isinstance(value, float) and (value != value or value == float('inf') or value == -float('inf')) else value) for key, value in record.items()}
                for record in data
            ]
        cleaned_data = clean_data(leader_info_select)
        return JSONResponse(content=cleaned_data)
        # return leader_info_select
    except Exception as e:
        return {"error": str(e), "traceback": repr(e)}
    # except Exception as e:
    #     return {"error": str(e)}
    
@app.get("/FanAttitude/{time}/{event_id}/{leader_id}")
async def fan_attitude(time:str, event_id: str, leader_id: str):
    """
    获取粉丝态度数据

    - time: 时间限制
    - event_id: event的id
    - leader_id: leader的id
    - 返回粉丝态度数据:积极人数、中立人数、消极人数。
    - 可实现leader_id：17980523，44196397，91583544，321954654，1319287761048723458
    """
    try:
        print(time)
        time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

        # Read the Excel file
        # df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_fan/17980523.csv')

        # 根据 leader_id 构造文件路径
        file_path = f'/root/autodl-tmp/syh/9_11_wwt/leader_fan/{leader_id}.csv'
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File for leader_id {leader_id} not found.")
        df = pd.read_csv(file_path)
        
        # Convert the 'createdb' column to datetime
        df['createdb'] = pd.to_datetime(df['createdb'])
        
        # Filter rows where 'createdb' is less than the given time limit
        time_limit = pd.to_datetime(time)
        df_filtered = df[df['createdb'] < time_limit].copy()  # Make a copy to avoid SettingWithCopyWarning
        
        # Define the bins and labels for the sentiment calculation
        bins = [-float('inf'), -0.2, 0.2, float('inf')]
        labels = ['negative', 'neutral', 'positive']
        
        # Use the `pd.cut` function to assign sentiment based on 'init_att' values
        df_filtered.loc[:, 'sentiment'] = pd.cut(df_filtered['init_att'], bins=bins, labels=labels, right=False)
        
        # Count the number of each sentiment
        sentiment_counts = df_filtered['sentiment'].value_counts()
        
        # Return the counts as a dictionary
        result = {
            'positive': sentiment_counts.get('positive', 0),
            'neutral': sentiment_counts.get('neutral', 0),
            'negative': sentiment_counts.get('negative', 0)
        }
        converted_dict = {key: str(value) for key, value in result.items()}
        
        return converted_dict
    
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/LeaderImpactIndex/{time}/{event_id}/{leader_id}")
async def leader_impact_index(time: str, leader_id: str, event_id: str):
    """
    获取leader的影响力

    - time: 当前日期2022-10-05
    - event_id: event的id
    - leader_id: leader的id
    - 返回leader的影响力:Pi, Ei, Ni;
    - 可实现leader_id：17980523，44196397，91583544，321954654，1319287761048723458
    """
        
    try:
        time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        print(time)
        # time_limit = datetime.strptime(time, '%Y-%m-%d')
        print(time_limit)
        # 使用pandas读取Excel文件

        # df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_influ/17980523.csv')
        file_path = f'/root/autodl-tmp/syh/9_11_wwt/leader_influ/{leader_id}.csv'
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File for leader_id {leader_id} not found.")
        df = pd.read_csv(file_path)

        df['createdb'] = pd.to_datetime(df['createdb'])
        # print(len(df))
        df_filtered = df[df['createdb'] < time_limit]
        # print(f"Filtered rows: {len(df_filtered)}")

        df_filtered_sorted = df_filtered.sort_values(by='createdb', ascending=True)
        # print(df_filtered_sorted)
        # 获取排序后的第一条记录
        first_row = df_filtered_sorted.tail(1)
        # print(first_row)
        # 检查所需的列是否存在
        required_columns = ['Pi','Ei', 'Ni']
        for column in required_columns:
            if column not in df.columns:
                print(f"The column '{column}' is missing in the file.")
                return None
        # 提取所需的列
        result = first_row[required_columns]

    except FileNotFoundError:
        print("The file was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    # result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # result['date'] = result['date'].dt.strftime('%Y-%m-%d')
    result1 = tuple(result.to_dict(orient='records'))
    return tuple(result1)
    

@app.get("/LeaderCompare/{time}/{event_id}/{leader_id}")
async def leader_compare(leader_id: str, time: str, event_id: str):
    """
    获取leader的对比数据

    - time: 当前日期2022-10-05
    - event_id: event的id
    - leader_id: leader的id
    - 返回leader的对比数据:Pi,Ei,Ni,PZ, DG, NCI
    - 可实现leader_id：17980523，44196397，91583544，321954654，1319287761048723458
    """
    try:
        # time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        print(time)
        time_limit = datetime.strptime(time, '%Y-%m-%d')
        print(time_limit)

        # 使用pandas读取Excel文件
        # df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_influ/17980523.csv')
        file_path = f'/root/autodl-tmp/syh/9_11_wwt/leader_influ/{leader_id}.csv'
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File for leader_id {leader_id} not found.")
        df = pd.read_csv(file_path)

        df['createdb'] = pd.to_datetime(df['createdb'])
        # print(len(df))
        df_filtered = df[df['createdb'] < time_limit]
        # print(f"Filtered rows: {len(df_filtered)}")

        df_filtered_sorted = df_filtered.sort_values(by='createdb', ascending=True)
        # print(df_filtered_sorted)
        # 获取排序后的第一条记录
        first_row = df_filtered_sorted.tail(1)
        # print(first_row)
        # 检查所需的列是否存在
        required_columns = ['Pi','Ei', 'Ni','PZ','DG','NCI']
        for column in required_columns:
            if column not in df.columns:
                print(f"The column '{column}' is missing in the file.")
                return None
        # 提取所需的列
        result = first_row[required_columns]

    except FileNotFoundError:
        print("The file was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    # result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # result['date'] = result['date'].dt.strftime('%Y-%m-%d')
    result1 = tuple(result.to_dict(orient='records'))
    return tuple(result1)
    

@app.get("/LeaderEvent/{time}/{event_id}/{leader_id}")
async def leader_event(time: str, event_id: str, leader_id: str):
    """
    获取leader的相关推文

    - time: 当前日期
    - event_id: event的id
    - leader_id: leader的id
    - 返回leader的相关推文数据:用户名, 用户id, 推文内容, 评论数, 转发数, 点赞数, 图片或视频, 发布时间
    - 可实现leader_id：17980523，44196397，91583544，321954654，1319287761048723458
     """
    # df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_influ/17980523.csv')
    file_path = f'/root/autodl-tmp/syh/9_11_wwt/leader_fan/{leader_id}.csv'
    if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File for leader_id {leader_id} not found.")
    df = pd.read_csv(file_path)

    # 将输入的时间字符串转换为datetime对象
    input_datetime = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    
    # 确保createdb列是datetime类型
    df['createdb'] = pd.to_datetime(df['createdb'])
    
    # 筛选出创建时间早于输入时间的条目
    filtered_df = df[df['createdb'] < input_datetime]
    
    # 重命名列以匹配所需的属性名
    filtered_df = filtered_df.rename(columns={
        'name': 'user_name',
        'uid': 'user_id',
        'content': 'content',
        'replycount': 'replycount',
        'retweetcount': 'retweetcount',
        'favoritecount': 'favoritecount',
        'createdb': 'create_time'
    })
    
    # # 将media列中的第一个URL提取出来作为picture
    # filtered_df['picture'] = filtered_df['picture'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
    
    # 选择需要的列
    result_df = filtered_df[['user_name', 'user_id', 'content', 'replycount', 'retweetcount', 'favoritecount', 'media', 'create_time','avatar']]

    result_df = result_df.replace([np.inf, -np.inf], None)  # 将 inf 和 -inf 替换为 None
    result_df = result_df.fillna('')  # 将 NaN 替换为 None

    # 将DataFrame转换为字典列表
    result_list = result_df.to_dict('records')
    
    return tuple(result_list)

@app.get("/LeaderEventCount/{time}/{event_id}/{leader_id}")
async def leader_event_count(time: str, event_id: str, leader_id: str):
    """
    统计leader点赞总人数、评论总人数、转发总人数随时间变化

    - time: 当前日期2022-10-05
    - event_id: event的id
    - leader_id: leader的id
    - 返回时间、点赞总人数、评论总人数、转发总人数
    - 可实现leader_id：17980523，44196397，91583544，321954654，1319287761048723458
    """
    # 尝试读取Excel文件
    try:
        time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        # time_limit = datetime.strptime(time, '%Y-%m-%d')
        # 使用pandas读取Excel文件
        # df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_influ/17980523.csv')
        file_path = f'/root/autodl-tmp/syh/9_11_wwt/leader_influ/{leader_id}.csv'
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File for leader_id {leader_id} not found.")
        df = pd.read_csv(file_path)

        df['createdb'] = pd.to_datetime(df['createdb'])
        df_filtered = df[df['createdb'] < time_limit]
        print(f"Filtered rows: {len(df_filtered)}")

        # 检查所需的列是否存在
        required_columns = ['createdb', 'favoritecount','retweetcount', 'replycount', ]
        for column in required_columns:
            if column not in df.columns:
                print(f"The column '{column}' is missing in the file.")
                return None
        
        # 提取所需的列
        result = df_filtered[required_columns]

    except FileNotFoundError:
        print("The file was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    column_mapping = {
    'createdb': 'time',
    'replycount': 'comment_count_plt',
    'retweetcount': 'like_count_plt',
    'favoritecount': 'repost_count_plt',
    
    }
    result = result.rename(columns=column_mapping)
    result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # result['time'] = result['time'].dt.strftime('%Y-%m-%d')
    result1 = tuple(result.to_dict(orient='records'))
    return tuple(result1)

@app.get("/LeaderImpactPredict/{time}/{event_id}/{leader_id}")
async def leader_impact_predict(time: str, event_id: str, leader_id: str):
    """
    获取leader的影响力以及预测影响力

    - time: 当前日期
    - event_id: event的id
    - leader_id: leader的id
    - 返回时间、leader此刻的影响力、未来时间、影响力预测值
    """
    try:
        time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        # time_limit = datetime.strptime(time, '%Y-%m-%d')
        # 使用pandas读取Excel文件
        # df = pd.read_csv('/root/autodl-tmp/syh/9_11_wwt/leader_influ/17980523.csv')
        file_path = f'/root/autodl-tmp/syh/9_11_wwt/leader_influ/{leader_id}.csv'
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File for leader_id {leader_id} not found.")
        df = pd.read_csv(file_path)
        
        print(f"Columns in the DataFrame: {df.columns}")

        df['createdb'] = pd.to_datetime(df['createdb'])
        df_filtered_0 = df[df['createdb'] < time_limit]
        print(f"Filtered rows: {len(df_filtered_0)}")
        df_filtered_1 = df[df['createdb'] >= time_limit]
        print(f"Filtered rows: {len(df_filtered_0)}")

        # 检查所需的列是否存在
        required_columns = ['createdb', 'infu']
        for column in required_columns:
            if column not in df.columns:
                print(f"The column '{column}' is missing in the file.")
                return None
        
        # 提取所需的列
        result_0 = df_filtered_0[required_columns]
        result_1 = df_filtered_1[required_columns]

    except FileNotFoundError:
        print("The file was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    column_mapping0 = {
    'createdb': 'time',
    'infu': 'leader_influence',
    
    }
    column_mapping1 = {
    'createdb': 'time',
    'infu': 'leader_influence_predict',
    
    }

    result0 = result_0.rename(columns=column_mapping0)
    result1 = result_1.rename(columns=column_mapping1)
    result0['time'] = result0['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # result0['time'] = result0['time'].dt.strftime('%Y-%m-%d')
    result00 = tuple(result0.to_dict(orient='records'))
    result1['time'] = result1['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # result1['time'] = result1['time'].dt.strftime('%Y-%m-%d')
    result11 = tuple(result1.to_dict(orient='records'))
    return  tuple(result00),tuple(result11)

@app.get("/LeaderSocialNet/{time}/{event_id}/{leader_id}")
async def leader_social_net(time: str, event_id: str, leader_id: str):
    """
    获取leader的社交网络

    - time: 当前日期    
    - event_id: event的id
    - leader_id: leader的id
    - 返回leader的社交传播网络,包含用户节点号、用户id、用户名、传播时刻、点亮状态;（网络对应的）时间、节点大小、节点连线
    """
    try:
        # 解析传入的时间字符串
        time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

        # 读取 Excel 文件并确保 'time' 列为 datetime 类型
        df = pd.read_excel('leader_net.xlsx')
        df['time'] = pd.to_datetime(df['time'])

        # 时间过滤
        df_filtered = df[df['time'] < time_limit]

        # 检查所需的列是否存在
        required_columns = ['user_id', 'leader_id', 'leader_name', 'time', 'status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Missing columns: {', '.join(missing_columns)}"}

        # 提取所需的列
        result = df_filtered[required_columns]

        # 重命名列
        column_mapping = {
            'leader_id': 'user_id',
            'leader_name': 'user_name',
        }
        result = result.rename(columns=column_mapping)

        # 处理重复的列名
        if result.columns.duplicated().any():
            print("Found duplicate columns, handling them...")
            # 删除重复的列，只保留第一次出现的列
            result = result.loc[:, ~result.columns.duplicated()]
            # 或者可以使用其他方法，如重命名重复列
            # result.columns = [f'{col}_{i}' if result.columns.tolist().count(col) > 1 else col
            #                    for i, col in enumerate(result.columns)]

        # 格式化时间列
        result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 转换为字典列表
        result1 = tuple(result.to_dict(orient='records'))

    except FileNotFoundError:
        return {"error": "The file was not found. Please check the file path."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

    # 读取社交网络数据
    try:
        with open('./follower_1w.json', 'r', encoding='utf-8') as file:
            following_info = json.load(file)
        
        # 构建社交网络图
        network = nx.DiGraph()
        network.add_nodes_from([a for a in following_info.keys()])
        network.add_edges_from([(a, b) for a in following_info.keys() for b in following_info[a]])

        # 计算图的度数和边
        network_dt1 = {
            "degrees": tuple(dict(network.degree()).values()), 
            "edges": tuple(network.edges())
        }

        networks_dt = {
            "2024-11-16 21:36:18": network_dt1,
            "2024-11-17 21:36:18": network_dt1,
            "2024-11-18 21:36:18": network_dt1,
            "2024-11-19 21:36:18": network_dt1
        }

    except FileNotFoundError:
        return {"error": "Social network data file not found."}
    except Exception as e:
        return {"error": f"An error occurred while loading network data: {str(e)}"}

    # 返回整合后的数据
    return {
        "leader_data": result1,
        "network_data": networks_dt
    }

@app.get("/NetworkInit/{json_data}")
async def NetworkInit(json_data: str):
    data  = eval(json_data)
    print(data)
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
        profile_path = './profiles_random_data.csv'
        if os.path.exists(profile_path):
            df = pd.read_csv(profile_path)
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
    返回网络数据
    
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
    TF_degree: bool, TF_att: bool, TF_pcomment: bool, TF_sample:bool, TF_leader: bool,
    TF_media: bool 对应用户过滤的条件
    
    cnt_dc: float, cnt_eg: float, cnt_t: float, cnt_l: float, cnt_sc: float, cnt_sd: float, 
    cnt_ai: float, cnt_ep: float, cnt_bm: float, cnt_c: float 对应需要操作的用户类型
    
    operation: str, edges: str  对边的删改进行操作
    
    """
    edge_list = item.edges.split(',')  # 将边字符串拆分成列表
    print(f"Edges to {item.operation}: {edge_list}")  # 输出边列表
    params = [item.cnt_dc, item.cnt_eg, item.cnt_t, item.cnt_l, item.cnt_sc, item.cnt_sd, item.cnt_ai, item.cnt_ep, item.cnt_bm, item.cnt_c]
    profile_path = './profiles_random_data.csv'
    if os.path.exists(profile_path):
        df = pd.read_csv(profile_path)
    else:
        raise FileNotFoundError("profiles_random_data.csv 文件不存在！")
    with open('./follower_1000.json', 'r', encoding='utf-8') as file:
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
    if sum(params) == 1:                                                    
        try:
            prof_color_mapping = {
                "Doctor": "Red",
                "Engineer": "Orange",
                "Teacher": "Yellow",
                "Lawyer": "Green",
                "Scientist": "Blue",
                "Software Developer": "Indigo",
                "Artificial Intelligence Specialist": "Violet",
                "Entrepreneur": "Black",
                "Business Manager": "Brown",
                "Consultant": "Gold"
            }
            print(df.shape[0])
            prof_distribution = {
                "Doctor": df.shape[0] * params[0],
                "Engineer": df.shape[0] * params[1],
                "Teacher": df.shape[0] * params[2],
                "Lawyer": df.shape[0] * params[3],
                "Scientist": df.shape[0] * params[4],
                "Software Developer": df.shape[0] * params[5],
                "Artificial Intelligence Specialist": df.shape[0] * params[6],
                "Entrepreneur": df.shape[0] * params[7],
                "Business Manager": df.shape[0] * params[8],
                "Consultant": df.shape[0] * params[9]
            }
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

@app.get("/NetworkOverview")
async def read_network_for_overview():
    """
    界面一的全局实时网络拓扑，返回整个网络的用户连接关系和意见态度。

    - 没有输入参数。
    - 返回一个list对象，每个元素是一个字典。字典包括用户节点id（userID）（用于跳转页面3）、用户类型（role）、用户颜色（color）、粉丝数量（fans_num）（用于设置节点大小）、相邻节点id列表（fans）。
    - 示例数据：[ {"userID": "1", "role":"activist" , "color": "green", "fans_num": 2, "fans":["2","3"]}, {...}, {...}, ... ]。

    """
    # 读取csv文件，将用户类型、颜色列转换为列表
    current_dir = os.path.dirname(os.path.abspath(__file__)) #  获取当前脚本文件所在目录的上一级目录
    csv_path = os.path.join(current_dir, "profiles_random_data.csv")
    df = pd.read_csv(csv_path)
    id_list = df['id'].tolist()
    role_list = df['roles'].tolist()
    color_list = df['colors'].tolist()
    # 读取 JSON 数据，并将其转换为 Python 的字典处理
    json_path = os.path.join(current_dir, "follower_1000.json")
    with open(json_path, "r") as f:
        data = json.load(f)
        # 读取各用户粉丝，以及数量
        fans = [ value for key,value in data.items()]
        fans_num = [len(i) for i in fans]
    network_data = []
    for i in range(len(id_list)):
        # 将粉丝列表中的id提取
        fansID = [ (re.findall('\d+', fan)[0]) for fan in fans[i] ]
        point_data = {"userID": str(id_list[i]),"role": role_list[i], "color": color_list[i], "fans_num": fans_num[i], "fans": fansID}
        network_data.append(point_data)
    return network_data

@app.get("/TopicCount")
async def read_topic_for_count():
    """
    界面一的事件统计，返回热门事件及其信息数量。

    - 没有输入参数。
    - 返回一个字典对象，键为事件名，值为信息数量。
    - 示例数据：{"选民登记问题": 2, "邮寄投票争议": 3,  ...}
    
    """
    # 使用测试数据TopicTwitter_test.csv
    df = pd.read_csv("TwitterTopic_data.csv")
    # 读取各主题的推文数量
    cnt_1 = len(df[df['topic']=='AI预言'])
    cnt_2 = len(df[df['topic']=='巴西总统选举舞弊'])
    cnt_3 = len(df[df['topic']=='新冠疫苗'])
    cnt_4 = len(df[df['topic']=='美国登月造假'])
    result = [
        {
            'name': 'AI预言',
            'value': cnt_1
        },
        {
            'name': '巴西总统选举舞弊',
            'value': cnt_2
        },
        {
            'name': '新冠疫苗',
            'value': cnt_3
        },
        {
            'name': '美国登月造假',
            'value': cnt_4
        }
    ]
    return result

@app.get("/AttiPercentage")
async def read_atti_for_draw():
    """"
    界面一的情感占比，返回正面、负面和中性信息的百分比。

    - 没有输入参数。 
    - 返回一个字典对象，键为情感名，值为所占百分比。
    - 示例数据：{"positive": 0.30, "negative": 0.30, "neutral":0.40 }
    
    """
    df = pd.read_csv("TwitterTopic_data.csv")
    cnt_pos = round( len(df[df["atti"]>0]) / len(df), 2)
    cnt_neg = round( len(df[df["atti"]<0]) / len(df), 2)
    cnt_neu = round( len(df[df["atti"]==0]) / len(df),2)
    data =  [
        { 'value': cnt_pos, 'name': 'positive', 'itemStyle': { 'color': '#2979ff' } },
        { 'value': cnt_neg, 'name': 'negative', 'itemStyle': { 'color': '#fde74c' } },
        { 'value': cnt_neu, 'name': 'neutral', 'itemStyle': { 'color': '#ff8a80' } }
      ]
    return data

@app.get("/WeekTrend")
async def read_week_trend(request: Request):
    """
    Retrieve the attitude scores of selected topics in the past week.
    
    - **request: Request**: User selected topic. 
        Note that the topic must appear in "@app.get("/TopicCount")". 
        example of the url: ../WeekTrend/?event=选民登记问题&event=邮寄投票争议
    - Return a dict object, which contains the weekly date list and the attitude scores of selected topics.

    """
    # 对获取到的事件名称转换为列表
    event_list = request.query_params.getlist('event')
    # 获取当天及前1周的时间列表
    today = datetime.now().date()
    date_list = [today - timedelta(days = i) for i in range(7)][::-1]
    formatted_date_list = [date.strftime("%m - %d") for date in date_list]
    # 随机生成事件一周的态度值列表
    trend_data = []
    for event in event_list:
        event_list.append({
            'name': event,
            'type': 'line',
            'stack': 'Total',
            'data': [round(random.uniform(-1,1),2) for _ in range(7)]
        })
    result = {"time": date_list, "trend": trend_data}
    return result

@app.get("/YearAnalysis")
async def read_year_analysis():
    """
    Retrieve the attitude scores of all topics in 12 months of the past two years.

    - No input parameters
    - Return a dict object, whose key is year, and value is the attitude score list.
    
    """
    # 随机生成年份的态度值
    result = {"2024": [round(random.uniform(-1,1),2) for _ in range(12)], "2023": [round(random.uniform(-1,1),2) for _ in range(12)]}
    return result

# 以下函数用于界面1的“全局统计信息”，暂时返回随机数, 后期需要根据数据库优化,
def cal_total(observe_time, data):
    # return len(data)
    if observe_time == "day":
        result = random.randint(50,100)
    elif observe_time == "week":
        result = random.randint(350,700)
    elif observe_time == "month":
        result = random.randint(1400,2800)
    else:
        result = random.randint(5000,6000)
    return result

def cal_ring_growth(observe_time, data):
    # current_data = data[observe_time]
    # last_data = data[observe_time -1]
    # result = (len(current_data) - len(last_data) ) / len(last_data)
    # return result*100
    return round(random.random(),2)

def cal_average(data):
    # result = sum(data[atti])/len(data)
    return round(random.random(),2)

def cal_bias(data):
    # result = sum(data[atti]-data[atti].mean())/len(data)
    return round(random.random(),2)
    
def cal_var(data):
    # result = sum((data[atti]-data[atti].mean())**2)/len(data)
    return round(random.random(),2)
    
def cal_NegTotal(observe_time, data):
    # result = sum(data[atti]<0)
    if observe_time == "day":
        result = random.randint(10,20)
    elif observe_time == "week":
        result = random.randint(200,300)
    elif observe_time == "month":
        result = random.randint(800,900)
    else:
        result = random.randint(2000,4000)
    return result

def cal_nonNegTotal(data):
    # result = sum(data[atti]>=0)
    return random.randint(1000,2000)

def cal_leader_num(data):
    # result = count(data[follower_number]>6)
    return random.randint(30,50)

# 舆情波动指数用极差来量化
def cal_fluc(data):
    # result = (data[atti].max()-data[atti].min())/10
    return round(random.random(),2)

# 舆情传播速率用节点数量变化率来量化
def cal_spread_rate(data):
    # result = (len(data) - len(old_data))/ len(old_data)
    return round(random.random(),2)

# 舆情转化率定义为（非负面舆情新增数量 - 负面舆情新增数量）/（初始舆情总量）
def cal_convert_rate(data):
    # result = (count(data[atti]>=0) - count(data[atti]<0)) / len(old_data)
    return round(random.random(),2)

# 信息增速，与环比定义相同
def cal_grow_rate(data):
    # current_data = data[observe_time]
    # last_data = data[observe_time -1]
    # result = (len(current_data) - len(last_data) ) / len(last_data)
    return round(random.random(),2)

@app.get("/Statistics/{observe_time}")
async def read_Statistics(observe_time: str):
    """
    Retrieve the Statistics of the posts in the observed time set by user.

    - observe_time: the time set by user, which can be "day", "week", "month", "year"
    - return a dict object which contains the followings:
        total: total number of the posts
        ring_growth: post number's growth rate compared to the last period of time
        average: average of attitude scores of posts
        bias: bias of attitude scores of posts
        var: variance of attitude scores of posts
        neg total: total number of negative posts
        non-neg total: total number of non-negative posts
        leader_num: number of leader users
        fluc: fluctuation of attitude scores of posts
        spread_rate: spread rate of topics
        convert_rate: convert rate of attitudes
        grow_rate: post number's grow rate (same as ring_growth)
    
    """
    data = None
    total = cal_total(observe_time, data)
    result = {
        "total": total,
        "ring_growth": cal_ring_growth(observe_time, data), 
        "average":cal_average(data), 
        "bias": cal_bias(data), 
        "var": cal_var(data), 
        "neg total": cal_NegTotal(observe_time, data), 
        "non-neg total": total-cal_NegTotal(observe_time, data), 
        "leader_num": cal_leader_num(data), 
        "fluc": cal_fluc(data), 
        "spread_rate": cal_spread_rate(data), 
        "convert_rate": cal_convert_rate(data), 
        "grow_rate": cal_grow_rate(data)
    }
    return result

@app.get("/NewMessage/{num}")
async def read_NewMessage(num: int):
    """
    Retrieve the latest posts.

    - **num**: number of posts to retrieve
    - return a list object which contains dict objects with each describing a post, including 

    """
    result=[]
    data = pd.read_csv("TwitterTopic_data.csv")
    # 将每一行数据转换为一个字典, 所有字典组成列表
    data_list = data.to_dict(orient="records")
    # print(type(data_list))
    # print(data_list[0])
    # 按时间排序
    data_time_order = sorted(data_list, key = lambda x: datetime.strptime(x["createdb"].split()[0], "%Y/%m/%d"), reverse = True)
    # sorted_df = pd.DataFrame(data_time_order)
    # print(f"总共有帖子数：{len(sorted_df)}")
    # sorted_df.to_csv("format_TimeOrder.csv", index=False)
    read_message = data_time_order[:num]
    for i in range(num):
       message_dict = {"orderID": str(i),"content":read_message[i]["content"], "att":read_message[i]["atti"], "time":read_message[i]["createdb"],"source":"Twitter"}
       result.append(message_dict)
    return result

@app.get("/HotMessage/{num}")
async def read_HotMessage(num: int):
    result = []
    data = pd.read_csv("TwitterTopic_data.csv")
    # 将每一行数据转换为一个字典, 所有字典组成列表
    data_list = data.to_dict(orient="records")
    data_hot_order = sorted(data_list, key = lambda x: x["replycount"] + x["retweetcount"] + x["favoritecount"], reverse = True)
    read_message = data_hot_order[:num]
    hot_list = [x["replycount"] + x["retweetcount"] + x["favoritecount"] for x in read_message]
    hot_list = [x / max(hot_list) for x in hot_list]
    for i in range(num):
       message_dict = {"orderID": str(i),"content":read_message[i]["content"], "att":read_message[i]["atti"], "time":read_message[i]["createdb"],"source":"Twitter", "hot_value":hot_list[i]}
       result.append(message_dict)
    return result

@app.post("/UserLoad")
async def read_UserLoad(request: Request):
    try:
        body = await request.json()
        data = []
        for i in range(len(body)):
            data_dict = {"userID": body[i]["userID"], "name": body[i]["name"], "fans_num": body[i]["fans_num"], "gender":"男","figure":"普通用户","birth":"2000-01-01","traits":"乐观、耐心、敢于冒险","hobbies":"唱、跳、rap、篮球","memory": {"type":"阅读","content":"欢迎来到微博！随时随地，发现新鲜事~"}}
            data.append(data_dict)
        data = pd.DataFrame.from_dict(data)
        # 暂时写入UserLoad.csv文件中，后续可以写入数据库
        data.to_csv("UserLoad.csv", mode="w", header=True, index=False)
        return JSONResponse(content={"state":True})
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)

@app.post("/UserAdd")
async def read_UserAdd(request:Request):
    try:
        body = await request.json()
        df = pd.read_csv("UserLoad.csv")
        body = [{"userID": str(len(df)), "name": body["name"], "fans_num": body["fans_num"], "gender":"男","figure":"普通用户","birth":"2000-01-01","traits":"乐观、耐心、敢于冒险","hobbies":"唱、跳、rap、篮球","memory": {"type":"阅读","content":"欢迎来到微博！随时随地，发现新鲜事~"}}]
        data = pd.DataFrame(body)
        data.to_csv("UserLoad.csv", mode="a", header=False, index=False)
        return JSONResponse(content={"state":True,"id": str(len(df))})
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code = 400)

@app.delete("/UserDelete/{userID}")
async def read_UserDelete(userID:str):
    try:
        df = pd.read_csv("UserLoad.csv")
        df["userID"] = df["userID"].astype(str)
        df = df[df["userID"] != userID]
        df.to_csv("UserLoad.csv", mode="w", header=True, index=False)
        return JSONResponse(content={"state":True})
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)

@app.put("/UserProfileSave")
async def read_UserProfileSave(request: Request):
    try:
        body = await request.json() 
        df = pd.read_csv("UserLoad.csv")
        # 以下修改指定行
        df["userID"] = df["userID"].astype(str)
        df.loc[df["userID"] == body["userID"], "name"] = body["name"]
        df.loc[df["userID"] == body["userID"], "gender"] = body["gender"]
        df.loc[df["userID"] == body["userID"], "figure"] = body["figure"]
        df.loc[df["userID"] == body["userID"], "birth"] = body["birth"]
        df.loc[df["userID"] == body["userID"], "traits"] = body["traits"]
        df.loc[df["userID"] == body["userID"], "hobbies"] = body["hobbies"]
        df.loc[df["userID"] == body["userID"], "memory"] = str(body["memory"])
        df.to_csv("UserLoad.csv", mode="w", header=True, index=False)
        return JSONResponse(content={"state":True})
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)

@app.get("/PrivateChat/{userID}/{num}")
async def read_PrivateChat(userID: str, num: int):
    try:
        currentdir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(currentdir, 'PrivateChat.json')
        with open(json_path,'r') as f:
            data = json.load(f)
            sample_data = random.sample(data, num)
        chat = [{"userID": userID,"chatID": str(i), "friendID": str(random.randint(1,20)), "content": sample_data[i]} for i in range(num)]
        return chat
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)
    
@app.get("/PublicPost/{userID}/{num}")
async def read_PublicPost(userID: str, num: int):
    try:
        currentdir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(currentdir, 'PublicPost.json')
        with open(json_path,'r') as f:
            data = json.load(f)
            sample_data = []
            for _ in range(num):
                topic = random.choice(["AI预言","巴西总统选举舞弊","新冠疫苗引发不孕","美国登月造假"])
                sample_data.append(random.choice(data[topic]))
        post = [{"userID": userID,"postID": str(i), "content": sample_data[i]} for i in range(num)]
        return post
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)


@app.get("/UserAttributeStatistics/{eventdesc}")
async def UserAttributeStatistics(eventdesc: str):
    """
    功能：统计不同用户属性下（包括性别，年龄，人种，受教育程度，社会角色，经济水平，社交能力）的群体对事件的支持中立反对意见分布
    输入参数：eventdesc: 事件描述
    返回值：一个列表，列表中每个元素是一个字典，字典的键是用户属性，值是对应的支持中立反对意见分布
    """
    result = pd.read_excel('/root/autodl-tmp/syh/8_10_mwb/attributes_distribution.xlsx')
    
    # 将每一行内容转换为字典格式,并将所有字典组成一个列表
    output = tuple(result.to_dict(orient='records'))
    
    return output


@app.get("/OpinionLeaderInfluence/{eventdesc}")
async def OpinionLeaderInfluence(eventdesc: str):
    """
    功能：统计事件中的意见领袖对事件的影响力，伴随时间变化
    输入参数：eventdesc: 事件描述
    返回值：一个列表，列表中每个元素是一个字典。name：bob，time1：350；name：bob，time2：300；
    """
    try:
        result = pd.read_excel('/root/autodl-tmp/syh/8_10_mwb/influence_data.xlsx')
        if result.empty:
            raise HTTPException(status_code=400, detail="Excel file is empty")
        if result.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Excel file has insufficient columns")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading Excel file: {e}")
    
    try:
        first_column = result.iloc[:, 0]
        other_columns = result.iloc[:, 1:]
        
        
        # 生成所有双列数据的组合
        combined_data = []
        for col in other_columns.columns:
            for i in range(len(first_column)):
                combined_data.append({first_column.name: first_column[i], col: other_columns[col][i]})
        
        return combined_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")

@app.get("/WordCloudAnalysis/{eventdesc}/{time_interval}")
async def WordCloudAnalysis(eventdesc: str,time_interval:str):
    """
    功能：统计与事件相关的关键词及词频
    输入参数：eventdesc: 事件描述，time_interval: 时间区间
    返回值：一个列表，列表中每个元素是一个字典。word：中国，frequency：100；word：美国，frequency：80；
    """
    result = pd.read_excel('/root/autodl-tmp/syh/8_10_mwb/word_cloud.xlsx')
    
    output = tuple(result.to_dict(orient='records'))
    
    return output

@app.get("/UserAttitudeDynamics/{eventdesc}/{time_scale}")
async def UserAttitudeDynamics(eventdesc: str,time_scale:str):
    """
    功能：统计用户支持中立反对态度随时间变化的动态数据
    输入参数：eventdesc: 事件描述，time_scale: 时间尺度
    返回值：一个列表，列表中每个元素是一个字典。time：2022-01-01，support：100，neutral：80，oppose：20；
    """
    result = pd.read_excel('/root/autodl-tmp/syh/8_10_mwb/attitude_data.xlsx')

    output = tuple(result.to_dict(orient='records'))

    return output

@app.get("/RealtimeData/{eventdesc}/{time}")
async def RealtimeData(eventdesc: str, time: str):
    """
    功能：根据输入的时间，返回该时间点的实时数据
    输入参数：eventdesc: 事件描述，time: 时间
    返回值：字典，
    {
  "时间": "2024-11-23T07:00:00",
  "Speed ": 85,
  "User Engagement": 40,
  "Attitude ": "05:03:01",
  "Information Cocoon Effect": 0.25,
  "Network Aggregation": 0.45,
  "Heat of the Subject": "Warm"
   }
    """
    Realtime_data = pd.read_excel('/root/autodl-tmp/syh/8_10_mwb/Realtime_data.xlsx')
    
    # 将时间列转换为datetime格式
    Realtime_data.iloc[:, 0] = pd.to_datetime(Realtime_data.iloc[:, 0])
    
    # 将输入的时间字符串转换为datetime格式
    input_time = datetime.strptime(time, "%Y-%m-%d %H:%M")
    
    # 查找与输入时间匹配的行
    matching_row = Realtime_data[Realtime_data.iloc[:, 0] == input_time]
    
    if matching_row.empty:
        raise HTTPException(status_code=404, detail="Time not found in the data")
    
    # 将匹配的行转换为字典格式
    result_dict = matching_row.to_dict(orient='records')[0]
    
    return result_dict

@app.get("/GeographicalAttitudes/{eventdesc}/{country}/{time}")
async def GeographicalAttitudes(eventdesc: str,country:str,time:str):
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
    geo_data = pd.read_excel('/root/autodl-tmp/syh/8_10_mwb/geographical_attitudes.xlsx')    
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

@app.get("/ForwardingTrends/{eventdesc}")
async def ForwardingTrends(eventdesc: str):
    """
    功能：统计事件不同时间的转发量
    输入：事件描述（eventdesc）
    返回值：一个列表，列表中每个元素是一个字典。time：2022-01-01，forwarding：100；
    """
    result = pd.read_excel('/root/autodl-tmp/syh/8_10_mwb/Forwarding_data.xlsx')
    
    output = tuple(result.to_dict(orient='records'))
    
    return output

@app.get("/KeyUsers/{eventdesc}/{ID}")
async def KeyUsers(eventdesc: str,ID:str):
    """
    功能：根据输入的事件描述和ID，返回该ID的姓名，头像，粉丝，转发内容等信息
    输入：事件描述（eventdesc），ID（str）
    返回值：一个字典，包含姓名，头像，粉丝，转发内容等信息
    """
    key_users_data = pd.read_excel('/root/autodl-tmp/syh/8_10_mwb/KeyUsers.xlsx')

    key_users_data['ID'] = key_users_data['ID'].astype(str).str.strip()
    ID = ID.strip()
    
    matching_row = key_users_data[key_users_data['ID'] == ID]
    
    if matching_row.empty:
        raise HTTPException(status_code=404, detail="ID not found in the data")
    
    result_dict = matching_row.to_dict(orient='records')[0]
    
    return result_dict

@app.get("/CriticalUserPaths/{eventdesc}")
async def CriticalUserPaths(eventdesc: str):
    """
    功能：根据输入的事件描述，返回该事件的相关关键用户传播路径
    输入：事件描述（eventdesc）
    返回值：一个字典，有nodes和edges两个键，分别对应节点和边的信息；节点信息包括id，name，avatar等，边信息包括源节点和目标节点
    """
    path_data = pd.read_excel('/root/autodl-tmp/syh/8_10_mwb/CriticalUserPaths_data.xlsx')
    
    G = nx.DiGraph()
    
    # 假设Excel文件有一列source和一列target，以及其他信息列
    for index, row in path_data.iterrows():
        source = row['Source']
        target = row['Target']
        
        # 添加源节点的属性
        G.add_node(source, name=row['Name'], avatar=row['Avatar'])
        
        if pd.notna(target):  # 检查目标是否为空
            # 将目标节点拆分为单个字符
            for t in target:
                G.add_node(t)  # 添加目标节点
                G.add_edge(source, t)
    
    # 构建结果字典
    result = {
        "nodes": [{**G.nodes[node], "id": node} for node in G.nodes],
        "edges": list(G.edges)
    }
    
    return result

