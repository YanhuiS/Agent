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


@app.get("/LeaderChoose/")
async def leader_choose():
    result = read_list()
    result1=result[['leader','leader_id']]
    leader_info = tuple(result1.to_dict(orient='records'))
    return leader_info

@app.get("/LeaderBasicInfo/{id}")
async def leader_basic_info(id: int):
    result = read_list()
    specific_id = id
    specific_row = result.loc[result['leader_id'] == specific_id]

    # # 使用query方法根据id查找特定的行
    # specific_id = id
    # specific_row = result.query('leader_id == @specific_id')

    required_columns = ['leader', 'leader_id','leader_fan_count', 'leader_follower_count',  'leader_post_count','leader_liked_count', 'leader_repost_count', 'leader_comment_count', 'leader_pic','sex']
    # leader_info1 = specific_row[required_columns]
    # leader_info_select = tuple(leader_info1.to_dict(orient='records'))
    leader_info_select = tuple(specific_row[required_columns].to_dict(orient='records'))

    return leader_info_select

@app.get("/TipPoint/{time}")
async def tip_point(time:str):
    try:
        # 将输入的时间字符串转换为datetime对象
        time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        print(f"Time limit: {time_limit}")
        
        # 使用pandas读取Excel文件
        df = pd.read_excel('test.xlsx')
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
        print(df_sorted)
        # 选取前十个结果
        # df_top10 = df_sorted.head(10)
        df_top10 = df_sorted.iloc[:10]
        print(df_top10)
        print(f"Top 10 rows: {len(df_top10)}")
        
        # 将结果转换为字典列表，以便JSONResponse可以序列化
        required_columns = ['name', 'uid','favourites_count', 'media', 'retweetcount', 'createdb']
        leader_info_select = tuple(df_top10[required_columns].to_dict(orient='records'))
        return leader_info_select
    
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/FanAttitude/{time}")
async def fan_attitude(time:str):
    try:
        # Read the Excel file
        df = pd.read_excel('test _attitude.xlsx')
        
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
    
@app.get("/LeaderImpactIndex/{id}/{time}")
async def leader_impact_index(id: int, time: str):
    time0 =time
    result = read_influence(time0)
    specific_id = id
    specific_row = result.loc[result['leader_id'] == specific_id]

    # # 使用query方法根据id查找特定的行
    # specific_id = id
    # specific_row = result.query('leader_id == @specific_id')

    required_columns = ['leader', 'leader_id','Pi','Ei', 'Ni']
    # leader_info1 = specific_row[required_columns]
    # leader_info_select = tuple(leader_info1.to_dict(orient='records'))
    leader_info_select = tuple(specific_row[required_columns].to_dict(orient='records'))

    return leader_info_select

# @app.get("/LeaderCompare/{id}")
# async def leader_compare(id: int):
    # result = read_influence()
    # specific_id = id
    # specific_row = result.loc[result['leader_id'] == specific_id]

    # # # 使用query方法根据id查找特定的行
    # # specific_id = id
    # # specific_row = result.query('leader_id == @specific_id')

    # required_columns = ['leader', 'leader_id','Pi','Ei', 'Ni','PZ','DG','NCI']
    # # leader_info1 = specific_row[required_columns]
    # # leader_info_select = tuple(leader_info1.to_dict(orient='records'))
    # leader_info_select = tuple(specific_row[required_columns].to_dict(orient='records'))

    # return leader_info_select

@app.get("/LeaderEvent/{time}")
async def leader_event(time: str):
    df = pd.read_excel('test_influence.xlsx')
    
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
        'media': 'picture',
        'createdb': 'create_time'
    })
    
    # 将media列中的第一个URL提取出来作为picture
    filtered_df['picture'] = filtered_df['picture'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
    
    # 选择需要的列
    result_df = filtered_df[['user_name', 'user_id', 'content', 'replycount', 'retweetcount', 'favoritecount', 'picture', 'create_time']]
    
    # 将DataFrame转换为字典列表
    result_list = result_df.to_dict('records')
    
    return tuple(result_list)

@app.get("/LeaderEventCount/{time}")
async def leader_event_count(time: str):
    # 尝试读取Excel文件
    try:
        # time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        time_limit = datetime.strptime(time, '%Y-%m-%d')
        # 使用pandas读取Excel文件
        df = pd.read_excel('leader_in_data.xlsx')
        df['date'] = pd.to_datetime(df['date'])
        df_filtered = df[df['date'] < time_limit]
        print(f"Filtered rows: {len(df_filtered)}")

        # 检查所需的列是否存在
        required_columns = ['date', 'all_comment_count','all_retweet_count', 'all_like_count', ]
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
    'date': 'time',
    'all_comment_count': 'comment_count_plt',
    'all_retweet_count': 'like_count_plt',
    'all_like_count': 'repost_count_plt',
    
    }
    result = result.rename(columns=column_mapping)
    # result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result['time'] = result['time'].dt.strftime('%Y-%m-%d')
    result1 = tuple(result.to_dict(orient='records'))
    return tuple(result1)


@app.get("/LeaderImpactPredict/{time}")
async def leader_impact_predict(time: str):
    try:
        # time_limit = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        time_limit = datetime.strptime(time, '%Y-%m-%d')
        # 使用pandas读取Excel文件
        df = pd.read_excel('leader_in_data.xlsx')
        df['date'] = pd.to_datetime(df['date'])
        df_filtered_0 = df[df['date'] < time_limit]
        print(f"Filtered rows: {len(df_filtered_0)}")
        df_filtered_1 = df[df['date'] > time_limit]
        print(f"Filtered rows: {len(df_filtered_0)}")

        # 检查所需的列是否存在
        required_columns = ['date', 'influence']
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
    
    column_mapping = {
    'date': 'time',
    'influence': 'leader_influence',
    
    }

    result0 = result_0.rename(columns=column_mapping)
    result1 = result_1.rename(columns=column_mapping)
    # result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result0['time'] = result0['time'].dt.strftime('%Y-%m-%d')
    result00 = tuple(result0.to_dict(orient='records'))
    result1['time'] = result1['time'].dt.strftime('%Y-%m-%d')
    result11 = tuple(result1.to_dict(orient='records'))
    return  tuple(result00),tuple(result11)


@app.get("/Network_init/{json_data}")
async def network_init(json_data: str):
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
            colors.append(profiles["color"][i])
            roles.append(profiles["role"][i])
            names.append(profiles["name"][i])
            genders.append(profiles["gender"][i])
            ages.append(profiles["age"][i])
            uids.append(profiles["uid"][i])

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
@app.get("/Network_show/")
async def Network_show():
    global network, att, likes, retweets, comments, views, favorites, colors, roles, names, genders, ages, uids,collects
    try:
        # 确保 network 已被初始化
        if network is None or len(network.nodes) == 0:
            return {"status": "error", "message": "Network has not been initialized."}

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
            "edges": tuple(network.edges())  # 图中的边
        }

        # 返回数据，FastAPI 会自动将其转换为 JSON 格式
        return network_data

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/Network_Manage/{para1}/{para2}/{para3}/{para4}/{para5}/{para6}/{para7}/{para8}/{para9}/{para10}/{para11}/{para12}/{para13}/{para14}/{para15}/{para16}/{para17}/{para18}/{para19}/{para20}/{para21}/{para22}/{para23}/{para24}/{para25}/{para26}/{para27}")
def Network_Manage(para1: bool, para2: bool, para3: bool, para4: bool, para5: bool, para6: bool, para7: bool, para8: bool, para9: bool, para10: bool,
                   para11: bool, para12: bool, para13: bool, para14: bool, para15: bool, para16: bool, para17: bool, para18: bool, para19: bool, para20: bool,
                   para21: bool, para22: bool, para23: bool, para24: bool, para25: bool, para26: bool, para27: bool,
                ):
    global network, att, likes, retweets, comments, views, colors, roles, names, genders, ages, uids, collects
    att, likes, retweets, comments, views, colors, roles, names, genders, ages, collects, uids = [], [], [], [], [], [], [], [], [], [], [], []
    if para1:
        try:
            # 读取 profiles_random_data.csv 文件
            profile_path = './profiles_random_data.csv'
            if os.path.exists(profile_path):
                df = pd.read_csv(profile_path)
            else:
                raise FileNotFoundError("profiles_random_data.csv 文件不存在！")
            # 筛选点赞数（likes）≥ 500 的用户
            filtered_df = df[df["likes"] >= 500]
            
            for _, row in filtered_df.iterrows():
                att.append(row["init_att"])
                likes.append(row["likes"])
                retweets.append(row["retweets"])
                comments.append(row["comments"])
                views.append(row["views"])
                collects.append(row["collects"])  # 确保favorites变量定义
                colors.append(row["color"])
                roles.append(row["role"])
                names.append(row["name"])
                genders.append(row["gender"])
                ages.append(row["age"])
                uids.append(row["uid"])
            filtered_node_ids = set(filtered_df["id"])  
            # 读取 follower_1000.json 文件并构建网络
            with open('./follower_1000.json', 'r', encoding='utf-8') as file:
                following_info = json.load(file)
            processed_following_info = {
                int(key[5:]): value  
                for key, value in following_info.items()
            }
            filtered_following_info = {key: processed_following_info[key] for key in processed_following_info if key in filtered_node_ids}
            reversed_following_info = {f"user_{key}": value for key, value in filtered_following_info.items()}
            network.clear()  # 确保这是正确的操作，如果不想清空所有节点，可以修改为删除特定节点
            network.add_nodes_from(reversed_following_info.keys())
            network.add_edges_from([(a, b) for a in reversed_following_info for b in reversed_following_info[a]])
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
            # 更新网络图，仅保留符合条件的节点和边
            network = network.subgraph(filtered_node_ids).copy()
            # 打印过滤后的网络数据，检查正确性
            return network_data
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    elif para2:
        try:
            # 读取 profiles_random_data.csv 文件
            profile_path = './profiles_random_data.csv'
            if os.path.exists(profile_path):
                df = pd.read_csv(profile_path)
            else:
                raise FileNotFoundError("profiles_random_data.csv 文件不存在！")
            # 筛选点赞数（likes）≥ 500 的用户
            filtered_df = df[df["retweets"] >= 50]
            for _, row in filtered_df.iterrows():
                att.append(row["init_att"])
                likes.append(row["likes"])
                retweets.append(row["retweets"])
                comments.append(row["comments"])
                views.append(row["views"])
                collects.append(row["collects"])  # 确保favorites变量定义
                colors.append(row["color"])
                roles.append(row["role"])
                names.append(row["name"])
                genders.append(row["gender"])
                ages.append(row["age"])
                uids.append(row["uid"])
            filtered_node_ids = set(filtered_df["id"])  
            # 读取 follower_1000.json 文件并构建网络
            with open('./follower_1000.json', 'r', encoding='utf-8') as file:
                following_info = json.load(file)
            processed_following_info = {
                int(key[5:]): value  
                for key, value in following_info.items()
            }
            filtered_following_info = {key: processed_following_info[key] for key in processed_following_info if key in filtered_node_ids}
            reversed_following_info = {f"user_{key}": value for key, value in filtered_following_info.items()}
            network.clear()  # 确保这是正确的操作，如果不想清空所有节点，可以修改为删除特定节点
            network.add_nodes_from(reversed_following_info.keys())
            network.add_edges_from([(a, b) for a in reversed_following_info for b in reversed_following_info[a]])
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
            # 更新网络图，仅保留符合条件的节点和边
            network = network.subgraph(filtered_node_ids).copy()
            # 打印过滤后的网络数据，检查正确性
            return network_data
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    elif para3:
        try:
            # 读取 profiles_random_data.csv 文件
            profile_path = './profiles_random_data.csv'
            if os.path.exists(profile_path):
                df = pd.read_csv(profile_path)
            else:
                raise FileNotFoundError("profiles_random_data.csv 文件不存在！")
            # 筛选点赞数（likes）≥ 500 的用户
            filtered_df = df[df["comments"] >= 100]
            for _, row in filtered_df.iterrows():
                att.append(row["init_att"])
                likes.append(row["likes"])
                retweets.append(row["retweets"])
                comments.append(row["comments"])
                views.append(row["views"])
                collects.append(row["collects"])  # 确保favorites变量定义
                colors.append(row["color"])
                roles.append(row["role"])
                names.append(row["name"])
                genders.append(row["gender"])
                ages.append(row["age"])
                uids.append(row["uid"])
            filtered_node_ids = set(filtered_df["id"])  
            # 读取 follower_1000.json 文件并构建网络
            with open('./follower_1000.json', 'r', encoding='utf-8') as file:
                following_info = json.load(file)
            processed_following_info = {
                int(key[5:]): value  
                for key, value in following_info.items()
            }
            filtered_following_info = {key: processed_following_info[key] for key in processed_following_info if key in filtered_node_ids}
            reversed_following_info = {f"user_{key}": value for key, value in filtered_following_info.items()}
            network.clear()  # 确保这是正确的操作，如果不想清空所有节点，可以修改为删除特定节点
            network.add_nodes_from(reversed_following_info.keys())
            network.add_edges_from([(a, b) for a in reversed_following_info for b in reversed_following_info[a]])
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
            # 更新网络图，仅保留符合条件的节点和边
            network = network.subgraph(filtered_node_ids).copy()
            # 打印过滤后的网络数据，检查正确性
            return network_data
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    elif para4:
        try:
            # 读取 profiles_random_data.csv 文件
            profile_path = './profiles_random_data.csv'
            if os.path.exists(profile_path):
                df = pd.read_csv(profile_path)
            else:
                raise FileNotFoundError("profiles_random_data.csv 文件不存在！")
            # 筛选点赞数（likes）≥ 500 的用户
            filtered_df = df[df["views"] >= 5000]
            for _, row in filtered_df.iterrows():
                att.append(row["init_att"])
                likes.append(row["likes"])
                retweets.append(row["retweets"])
                comments.append(row["comments"])
                views.append(row["views"])
                collects.append(row["collects"])  # 确保favorites变量定义
                colors.append(row["color"])
                roles.append(row["role"])
                names.append(row["name"])
                genders.append(row["gender"])
                ages.append(row["age"])
                uids.append(row["uid"])
            filtered_node_ids = set(filtered_df["id"])  
            # 读取 follower_1000.json 文件并构建网络
            with open('./follower_1000.json', 'r', encoding='utf-8') as file:
                following_info = json.load(file)
            processed_following_info = {
                int(key[5:]): value  
                for key, value in following_info.items()
            }
            filtered_following_info = {key: processed_following_info[key] for key in processed_following_info if key in filtered_node_ids}
            reversed_following_info = {f"user_{key}": value for key, value in filtered_following_info.items()}
            network.clear()  # 确保这是正确的操作，如果不想清空所有节点，可以修改为删除特定节点
            network.add_nodes_from(reversed_following_info.keys())
            network.add_edges_from([(a, b) for a in reversed_following_info for b in reversed_following_info[a]])
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
            # 更新网络图，仅保留符合条件的节点和边
            network = network.subgraph(filtered_node_ids).copy()
            # 打印过滤后的网络数据，检查正确性
            return network_data
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    elif para5:
        try:
            # 读取 profiles_random_data.csv 文件
            profile_path = './profiles_random_data.csv'
            if os.path.exists(profile_path):
                df = pd.read_csv(profile_path)
            else:
                raise FileNotFoundError("profiles_random_data.csv 文件不存在！")
            # 筛选点赞数（likes）≥ 500 的用户
            filtered_df = df[df["collects"] >= 70]
            for _, row in filtered_df.iterrows():
                att.append(row["init_att"])
                likes.append(row["likes"])
                retweets.append(row["retweets"])
                comments.append(row["comments"])
                views.append(row["views"])
                collects.append(row["collects"])  # 确保favorites变量定义
                colors.append(row["color"])
                roles.append(row["role"])
                names.append(row["name"])
                genders.append(row["gender"])
                ages.append(row["age"])
                uids.append(row["uid"])
            filtered_node_ids = set(filtered_df["id"])  
            # 读取 follower_1000.json 文件并构建网络
            with open('./follower_1000.json', 'r', encoding='utf-8') as file:
                following_info = json.load(file)
            processed_following_info = {
                int(key[5:]): value  
                for key, value in following_info.items()
            }
            filtered_following_info = {key: processed_following_info[key] for key in processed_following_info if key in filtered_node_ids}
            reversed_following_info = {f"user_{key}": value for key, value in filtered_following_info.items()}
            network.clear()  # 确保这是正确的操作，如果不想清空所有节点，可以修改为删除特定节点
            network.add_nodes_from(reversed_following_info.keys())
            network.add_edges_from([(a, b) for a in reversed_following_info for b in reversed_following_info[a]])
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
            # 更新网络图，仅保留符合条件的节点和边
            network = network.subgraph(filtered_node_ids).copy()
            # 打印过滤后的网络数据，检查正确性
            return network_data
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    elif para6:
        try:
            # 读取 profiles_random_data.csv 文件
            profile_path = './profiles_random_data.csv'
            if os.path.exists(profile_path):
                df = pd.read_csv(profile_path)
            else:
                raise FileNotFoundError("profiles_random_data.csv 文件不存在！")
            random_sample = df.sample(n=500, random_state=42)
            print(random_sample)
            for _, row in random_sample.iterrows():
                att.append(row["init_att"])
                likes.append(row["likes"])
                retweets.append(row["retweets"])
                comments.append(row["comments"])
                views.append(row["views"])
                collects.append(row["collects"])  # 确保favorites变量定义
                colors.append(row["color"])
                roles.append(row["role"])
                names.append(row["name"])
                genders.append(row["gender"])
                ages.append(row["age"])
                uids.append(row["uid"])
            filtered_node_ids = set(random_sample["id"])  
            print(filtered_node_ids)
            # 读取 follower_1000.json 文件并构建网络
            with open('./follower_1000.json', 'r', encoding='utf-8') as file:
                following_info = json.load(file)
            processed_following_info = {
                int(key[5:]): value  
                for key, value in following_info.items()
            }
            filtered_following_info = {key: processed_following_info[key] for key in processed_following_info if key in filtered_node_ids}
            reversed_following_info = {f"user_{key}": value for key, value in filtered_following_info.items()}
            network.clear()  # 确保这是正确的操作，如果不想清空所有节点，可以修改为删除特定节点
            network.add_nodes_from(reversed_following_info.keys())
            network.add_edges_from([(a, b) for a in reversed_following_info for b in reversed_following_info[a]])
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
            # 更新网络图，仅保留符合条件的节点和边
            network = network.subgraph(filtered_node_ids).copy()
            # 打印过滤后的网络数据，检查正确性
            return network_data
        except Exception as e:
            # 捕获异常并返回错误信息
            print("Error initializing network:", str(e))
            return {"status": "error", "message": str(e)}
    return network
# @app.get("/Network_Change/")
# async def Network_Change():