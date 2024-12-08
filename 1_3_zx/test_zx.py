from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
from datetime import datetime, timedelta
import random
import pandas as pd
import os
import re


app = FastAPI()
    
# git push测试
@app.get("/NetworkOverview") # 生成数据
async def read_network_for_overview():
    """
    Retrieve the Network Overview data.
    
    - No input parameter.
    - Return a list object, each element is a dict object containing userID, role, color, fans number, and fans of all users in Network.

    """
    # 读取csv文件，将用户类型、颜色列转换为列表
    current_dir = os.path.dirname(os.path.abspath(__file__)) #  获取当前脚本文件所在目录的上一级目录
    csv_path = os.path.join(current_dir, "profile_1000.csv")
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

@app.get("/TopicCount") # 推特数据
async def read_topic_for_count():
    """
    Retreive the count of each topic in the database.

    - No input parameters
    - Return a dict object, which contains the count of each topic.

    """
    # 使用测试数据TopicTwitter_test.csv
    df = pd.read_csv("TwitterTopic_data.csv")
    # 读取各主题的推文数量
    cnt_1 = len(df[df['topic']=='Russia Interference'])
    cnt_2 = len(df[df['topic']=='Trump Election'])
    cnt_3 = len(df[df['topic']=='Xinjiang Cotton'])
    result = [
        {
            "name": "通俄门",
            "value": cnt_1
        },
        {
            "name": "特朗普选举",
            "value": cnt_2
        },
        {
            "name": "新疆棉事件",
            "value": cnt_3
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
        example of the url: ../WeekTrend/?event=通俄门&event=特朗普选举
    - Return a dict object, which contains the weekly date list and the attitude scores of selected topics.

    """
    # 对获取到的事件名称转换为列表
    weektrend = []
    event_list = request.query_params.getlist('event')
    # 获取当天及前1周的时间列表
    today = datetime.now().date()
    date_list = [today - timedelta(days = i) for i in range(7)][::-1]
    # formatted_date_list = [date.strftime("%m - %d") for date in date_list]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 读取 csv 数据
    csv_path = os.path.join(current_dir, "TwitterTopic_data.csv")
    df = pd.read_csv(csv_path)
    # 随机生成事件一周的态度值列表
    for event in event_list:
        if event == "通俄门":
            topic = 'Russia Interference'
        elif event == "特朗普选举":
            topic = 'Trump Election'
        elif event == "新疆棉事件":
            topic = 'Xinjiang Cotton'
        df_tmp = df[df['topic'] == topic]
        if df_tmp.empty:
            weektrend.append({
                'name': event,
                'type': 'line',
                'stack': 'Total',
                'data':  'None'
            })
        else:
            dates = df_tmp['createdAt']
            format_str = '%Y/%m/%d %H:%M:%S'
            dt_list = [datetime.strptime(date, format_str) for date in dates]
            df_tmp['createdAt'] = dt_list
            is_datetime = all(isinstance(i, datetime) for i in df_tmp['createdAt'])
            # print(f"检查df日期列值是否全为datetime对象：{is_datetime}")
            end_time = df_tmp['createdAt'].max()
            is_datetime = isinstance(end_time, datetime)
            # print(f"检查end_time是否为datetime对象：{is_datetime}")
            atti_list = []
            for i in range(7):
                end_time_tmp = end_time - timedelta(days=i)
                start_time_tmp = end_time_tmp - timedelta(days=1)
                tmp = df_tmp[(df_tmp['createdAt'] >= start_time_tmp) & (df_tmp['createdAt'] <= end_time_tmp)]['atti'].mean()
                atti_list.append(round(tmp,2))
            weektrend.append({
                'name': event,
                'type': 'line',
                'stack': 'Total',
                'data':  atti_list[::-1]
            })
    result = {"time": date_list, "trend": weektrend}
    return result


@app.get("/WeekAnalysis") # 推特数据
async def read_week_analysis():
    """
    Retrieve the attitude scores of all topics in the past 2 weeks.

    - No input parameters
    - Return a dict object, whose key is year, and value is the attitude score list.
    
    """
    # 随机生成年份的态度值
    # result = {"2024": [round(random.uniform(-1,1),2) for _ in range(12)], "2023": [round(random.uniform(-1,1),2) for _ in range(12)]}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 读取 csv 数据
    csv_path = os.path.join(current_dir, "TwitterTopic_data.csv")
    df = pd.read_csv(csv_path)
    # 随机生成事件一周的态度值列表(这里选取最新的特朗普选举事件)
    df = df[df['topic'] == 'Trump Election']
    date_list = df['createdAt']
    format_str = '%Y/%m/%d %H:%M:%S'
    dt_list = [datetime.strptime(date, format_str) for date in date_list]
    df['createdAt'] = dt_list
    end_time = df['createdAt'].max()  
    atti_list1 = []
    atti_list2=[]
    for i in range(14):
        end_time_tmp = end_time - timedelta(days=i)
        start_time_tmp = end_time_tmp - timedelta(days=1)
        # print(f"第{i}轮start_time：{start_time_tmp}, end_time:{end_time_tmp}")
        tmp = df[(df['createdAt'] >= start_time_tmp) & (df['createdAt'] <= end_time_tmp)]['atti'].mean()
        if i < 7:
            atti_list1.append(round(tmp,2))
        else:
            atti_list2.append(round(tmp,2))
    result = {"this week": atti_list1[::-1], "last week": atti_list2[::-1]}
    return result

# 以下函数用于界面1的“全局统计信息”，暂时返回随机数, 后期需要根据数据库优化,
def cal_total(df, end_time, start_time):
    result = len(df[ (df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ])
    return result

def cal_ring_growth(df, deltatime, end_time, start_time):
    len1 = len(df[ (df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ])
    len2 = len(df[ (df['createdAt'] >= start_time - timedelta(days=deltatime)) & (df['createdAt'] <= start_time) ])
    result = (len1 - len2) / len2
    return round(result,2)

def cal_average(df, end_time, start_time):
    # result = sum(data[atti])/len(data)
    df = df[(df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ]
    result = df['atti'].mean()
    return round(result,2)

def cal_bias(df, end_time, start_time):
    # result = sum(data[atti]-data[atti].mean())/len(data)
    df = df[(df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ]
    mean = cal_average(df, end_time, start_time)
    dff = df['atti'] - mean
    result = sum([abs(x) for x in dff])/len(df)
    return round(result,4)

def cal_var(df, end_time, start_time):
    # result = sum((data[atti]-data[atti].mean())**2)/len(data)
    df = df[(df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ]
    mean = cal_average(df, end_time, start_time) 
    result = sum((df['atti']-mean)**2)/len(df)
    return round(result,4)
    
def cal_NegTotal(df, end_time, start_time): # ok
    # result = sum(data[atti]<0)
    df = df[(df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ]
    return sum(df['atti']<0)

def cal_NNegTotal(df, end_time, start_time): # ok
    # result = sum(data[atti]>=0)
    df = df[(df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ]
    return sum(df['atti']>= 0)

def cal_leader_num(df, end_time, start_time):
    # result = count(data[follower_number]>6)
    df = df[(df['createdAt'] >= start_time) & (df['createdAt'] <= end_time)]
    return sum(df['author/followers'] > 1000000)

# 舆情波动指数用“正反面态度人数的环比值”来量化，环比值越大，波动越大
def cal_fluc(df, deltatime, end_time, start_time):
    #cal_fluc(df, deltatime, end_time, start_time): ()  #  (这里需要连续2个月的数据，等王师兄的数据有了再写)
    df1 = df[(df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ]
    posnum1 = sum(df1['atti']>=0)
    negnum1 = sum(df1['atti']<0)
    df2 = df[(df['createdAt'] >= start_time - timedelta(days =deltatime)) & (df['createdAt'] <= start_time) ]
    posnum2 = sum(df2['atti']>=0)
    negnum2 = sum(df2['atti']<0)
    result = 0.5*(abs(posnum1-posnum2)/posnum2 + abs(negnum1-negnum2)/negnum2)
    return round(result,2)

# 舆情传播速率用节点数量变化率来量化
def cal_spread_rate(df, deltatime, end_time, start_time):
    # result = (len(data) - len(old_data))/ len(old_data)
    num1 = len(df[ (df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ])
    num2 = len(df[ (df['createdAt'] >= start_time - timedelta(days = deltatime)) & (df['createdAt'] <= start_time) ])
    result = (num1-num2)/num2
    return round(result,2)

# 舆情转化率定义为（非负面舆情新增数量 - 负面舆情新增数量）/（初始舆情总量）
def cal_convert_rate(df, deltatime, end_time, start_time):
    # result = (count(data[atti]>=0) - count(data[atti]<0)) / len(old_data)
    df1 = df[(df['createdAt'] >= start_time) & (df['createdAt'] <= end_time) ]
    posnum1 = sum(df1['atti']>=0)
    df2 = df[(df['createdAt'] >= start_time - timedelta(days =deltatime)) & (df['createdAt'] <= start_time) ]
    posnum2 = sum(df2['atti']>=0)
    print(f"posnum1:{posnum1}, posnum2:{posnum2}")
    result = (posnum1/len(df1) - posnum2/len(df2))/ (posnum2/len(df2))
    return round(result,2)

# 信息增速，与环比定义相同
def cal_grow_rate(df, deltatime, end_time, start_time):
    return cal_ring_growth(df, deltatime, end_time, start_time)

@app.get("/Statistics") 
async def read_Statistics(request: Request):
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
        spread_rate: rate of people proportion talking the topic
        convert_rate: convert rate of attitudes
        grow_rate: post number's grow rate (same as ring_growth)
    
    """

    # 把数据转换为datetime
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 读取 csv 数据
    csv_path = os.path.join(current_dir, "TwitterTopic_data.csv")
    df = pd.read_csv(csv_path)
    date_list = df['createdAt']
    format_str = '%Y/%m/%d %H:%M:%S'
    dt_list = [datetime.strptime(date, format_str) for date in date_list]
    df['createdAt'] = dt_list
    end_time = df['createdAt'].max()
    observe_time = request.query_params.get("observe_time")
    if observe_time == "day": deltatime = 1
    elif observe_time == "week": deltatime = 6
    else: deltatime = 29
    start_time = end_time - timedelta(days=deltatime)
    result = {
        "total": cal_total(df, end_time, start_time),
        "ring_growth": cal_ring_growth(df, deltatime, end_time, start_time), 
        "average":cal_average(df, end_time, start_time), 
        "bias": cal_bias(df, end_time, start_time), 
        "var": cal_var(df, end_time, start_time), 
        "neg_total": cal_NegTotal(df, end_time, start_time), 
        "non_neg_total": cal_NNegTotal(df, end_time, start_time), 
        "leader_num": cal_leader_num(df, end_time, start_time), 
        "fluc": cal_fluc(df, deltatime, end_time, start_time), 
        "spread_rate": cal_spread_rate(df, deltatime, end_time, start_time), 
        "convert_rate": cal_convert_rate(df, deltatime, end_time, start_time), 
        "grow_rate": cal_grow_rate(df, deltatime, end_time, start_time)
    }
    return result

@app.get("/NewMessage")
async def read_NewMessage(request: Request):
    """
    Retrieve the latest posts.

    - **num**: number of posts to retrieve
    - return a list object which contains dict objects with each describing a post, including 
    new order, content, post attitude, post time, post source 
    """
    result=[]
    data = pd.read_csv("TwitterTopic_data.csv")
    # 将每一行数据转换为一个字典, 所有字典组成列表
    data_list = data.to_dict(orient="records")
    # print(type(data_list))
    # print(data_list[0])
    # 按时间排序
    data_time_order = sorted(data_list, key = lambda x: datetime.strptime(x["createdAt"].split()[0], "%Y/%m/%d"), reverse = True)
    # sorted_df = pd.DataFrame(data_time_order)
    # print(f"总共有帖子数：{len(sorted_df)}")
    # sorted_df.to_csv("format_TimeOrder.csv", index=False)
    num = int(request.query_params.get("num"))
    read_message = data_time_order[:num]
    for i in range(num):
       message_dict = {"orderID": str(i),"content":read_message[i]["text"], "att":read_message[i]["atti"], "time":read_message[i]["createdAt"],"source":"Twitter"}
       result.append(message_dict)
    return result

@app.get("/HotMessage")
async def read_HotMessage(request: Request):
    """
    Retrieve the most popular posts
    
    -**num**: number of posts to retrieve
    - return a list object which contains dict objects with each describing a post including
    popular order, content, post attitude, post time, post source 
    """
    result = []
    data = pd.read_csv("TwitterTopic_data.csv")
    # 将每一行数据转换为一个字典, 所有字典组成列表
    data_list = data.to_dict(orient="records")
    data_hot_order = sorted(data_list, key = lambda x: x["replycount"] + x["retweetcount"] + x["favoritecount"], reverse = True)
    num = int(request.query_params.get("num"))
    read_message = data_hot_order[:num]
    hot_list = [x["replycount"] + x["retweetcount"] + x["favoritecount"] for x in read_message]
    hot_list = [x / max(hot_list) for x in hot_list]
    for i in range(num):
       message_dict = {"orderID": str(i),"content":read_message[i]["text"], "att":read_message[i]["atti"], "time":read_message[i]["createdAt"],"source":"Twitter", "hot_value":hot_list[i]}
       result.append(message_dict)
    return result

@app.get("/RetrieveUser") 
async def read_RetrieveUser(request: Request): 
    """
    Retrieve a user's id, and it's neighbours'id

    -"username": the name of the retrieved user
    - return a dict obj that contains username, user id and its neighbours id
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "UserLoad_1000.csv")
    df = pd.read_csv(csv_path)
    username = request.query_params.get("username")
    # 获取用户名对应id
    try:
        id = df.loc[(df["name"]) == username, "id"]
        id = int(id.values[0])
        json_path = os.path.join(current_dir, "follower_1000.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
            # 获取id对应邻居id
            neighbour = data[f"user_{id}"]
            neighbour_id = [ (re.findall('\d+', fan)[0]) for fan in neighbour]
        result = {"username": username, "user_id":str(id), "neighbour_id": neighbour_id}
        return result
    except Exception:
        return {"state": "No such user"}
    
@app.get("/UserTopology") # 推特数据
async def read_UserTopology(request: Request):
    """
    Retrieve a topology of a user, namely its neighbours and neighbours' neighbours

    -"userID": the id of the retrieved user
    -return a dict object which contains the following keys:
        -"userID": the id of the retrieved user
        -"layer1": the id of the neighbour
        -"layer2": the id of the neighbour's neighbour
    """
    userID = request.query_params.get("userID")
    result = {"userID": userID}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 读取 JSON 数据，并将其转换为 Python 的字典处理
    json_path = os.path.join(current_dir, "follower_1000.json")
    with open(json_path, "r") as f:
        data = json.load(f)
        fans = data[f"user_{userID}"]
        fansID = [ (re.findall('\d+', fan)[0]) for fan in fans]
        result["layer1"] = fansID
        # 将粉丝列表中的id提取
        layer2 = {}
        for fan in fansID:
            fansfans = data[f"user_{fan}"]
            fansfansID = [ (re.findall('\d+',fansfan)[0]) for fansfan in fansfans]
            layer2[fan] = fansfansID
        result["layer2"] = layer2
    return result

@app.get("/Userlist") # 生成数据
async def read_Userlist(request: Request):
    """
    Read user list
    
    - no input parameter
    - return a list object which contains dict objects whose keys are:
        -"id": id of the user
        -"name": name of the user
        -"fansNum": fans number of the user
    """
    try:
        pageNo = int(request.query_params.get("pageNo"))
        pageSize = int(request.query_params.get("pageSize"))
        maxnum = 1000 # 数据库中用户最大数量
        maxpageNo = maxnum // pageSize # 最大页数
        if pageNo > maxpageNo:
            return JSONResponse(content={"state":False, "error": "pageNo out of range"}, status_code=400)
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, "UserLoad_1000.csv")
            df = pd.read_csv(csv_path)
            # 获取页面范围用户id及其名字
            ids = df["id"].tolist()[(pageNo-1)*pageSize:pageNo*pageSize]
            names = df["name"].tolist()[(pageNo-1)*pageSize:pageNo*pageSize]
            # 获取该用户的粉丝数
            json_path = os.path.join(current_dir, "follower_1000.json")
            with open(json_path, "r") as f:
                data = json.load(f)
                FansNums = [len(data[f"user_{id}"]) for id in ids]
            result = [{"userID":ids[i], "name":names[i], "fansNum": FansNums[i]} for i in range(len(ids))]
            return result
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)

@app.get("/UserInfo") # 生成数据
async def read_UserInfo(request:Request):
    """
    Retreive a user info

    - "userID": id of the user
    - return a dict object whose keys are: id,name,gender,status,traits,interest,birth,memory    
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "UserLoad_1000.csv")
    df = pd.read_csv(csv_path)
    # 获取该用户的各个信息
    userID = request.query_params.get("userID")
    info = df.iloc[int(userID)].to_dict()
    info['id'] = userID 
    return info

@app.put("/UserProfileSave") # 生成数据
async def read_UserProfileSave(request: Request):
    """
    Save profile change

    - input is the json object of the user profile
    e.g 
    data_change = {"userID":"0", "name": "JasmineTurtle" , "gender": "female", ...}
    response = requests.put(url_4, json=data_change)

    - return state of "True" meaning the change has been saved in "UserLoad_1000.csv"
    """
    try:
        body = await request.json() 
        df = pd.read_csv("UserLoad_1000.csv")
        # 以下修改指定行
        df["id"] = df["id"].astype(str)
        df.loc[df["id"] == body["userID"], "name"] = body["name"]
        df.loc[df["id"] == body["userID"], "gender"] = body["gender"]
        df.loc[df["id"] == body["userID"], "status"] = body["status"]
        df.loc[df["id"] == body["userID"], "traits"] = body["traits"]
        df.loc[df["id"] == body["userID"], "interest"] = body["interest"]
        df.loc[df["id"] == body["userID"], "birth"] = body["birth"]
        df.loc[df["id"] == body["userID"], "memory"] = str(body["memory"])
        df.to_csv("UserLoad_1000.csv", mode="w", header=True, index=False)
        return JSONResponse(content={"state":True})
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)

@app.get("/PrivateChat") # 生成数据
async def read_PrivateChat(request: Request):
    """"
    Retrieve private chat of a user

    - **userID**: id of user, **num**: number of chats to retrieve
    - return a list object that contains dict object whose keys are: userID, chatID, friendID, content
    
    """
    try:
        userID = request.query_params.get("userID")
        # num = int(request.query_params.get("num"))
        pageNo = int(request.query_params.get("pageNo"))
        pageSize = int(request.query_params.get("pageSize"))
        maxnum = 50 # 数据库中对话数据的最大条数
        maxpageNo = maxnum // pageSize # 最大页数
        topic_list = request.query_params.getlist('topic')
        if pageNo > maxpageNo:
            return JSONResponse(content={"state":False, "error": "pageNo out of range"}, status_code=400)
        else:
            currentdir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(currentdir, 'PrivateChat.json')
            with open(json_path,'r') as f:
                data = json.load(f)
                # sample_data = random.sample(data, pageSize)
                flag = 1
                while flag:
                    friendID = str(random.randint(0,1000))
                    if friendID != userID:
                        flag = 0
                sample_data = []
                for _ in range(pageSize):
                    topic = random.choice(topic_list)
                    sample_data.append(random.choice(data[topic]))                    
                chat = [{"userID": userID,"chatID": str((pageNo-1)*pageSize+i+1), "friendID": friendID, "content": sample_data[i]} for i in range(pageSize)]
                return chat
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)
    
@app.get("/PublicPost") # 生成数据
async def read_PublicPost(request: Request):
    """
    Retrieve public post a user

    - **userID**: id the user, **num**: number of posts
    - return a list object that contains dict object whose keys are: userID, postID, content
    
    """
    try:
        userID = request.query_params.get("userID")
        # num = int(request.query_params.get("num"))
        pageNo = int(request.query_params.get("pageNo"))
        pageSize = int(request.query_params.get("pageSize"))
        maxnum = 50 # 数据库中对话数据的最大条数
        maxpageNo = maxnum // pageSize # 最大页数
        topic_list = request.query_params.getlist('topic')
        if pageNo > maxpageNo:
            return JSONResponse(content={"state":False, "error": "pageNo out of range"}, status_code=400)
        else:
            currentdir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(currentdir, 'PublicPost.json')
            with open(json_path,'r') as f:
                data = json.load(f)
                sample_data = []
                for _ in range(pageSize):
                    topic = random.choice(topic_list)
                    sample_data.append(random.choice(data[topic]))
            post = [{"userID": userID,"postID": str((pageNo-1)*pageSize+i+1), "content": sample_data[i]} for i in range(pageSize)]
            return post
    except Exception as e:
        return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)
    
# @app.post("/UserLoad")
# async def read_UserLoad(request: Request):
#     try:
#         body = await request.json()
#         data = []
#         for i in range(len(body)):
#             data_dict = {"userID": body[i]["userID"], "name": body[i]["name"], "fans_num": body[i]["fans_num"], "gender":"男","figure":"普通用户","birth":"2000-01-01","traits":"乐观、耐心、敢于冒险","hobbies":"唱、跳、rap、篮球","memory": {"type":"阅读","content":"欢迎来到微博！随时随地，发现新鲜事~"}}
#             data.append(data_dict)
#         data = pd.DataFrame.from_dict(data)
#         # 暂时写入UserLoad.csv文件中，后续可以写入数据库
#         data.to_csv("UserLoad.csv", mode="w", header=True, index=False)
#         return JSONResponse(content={"state":True})
#     except Exception as e:
#         return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)

# @app.post("/UserAdd")
# async def read_UserAdd(request:Request):
#     try:
#         body = await request.json()
#         df = pd.read_csv("UserLoad.csv")
#         body = [{"userID": str(len(df)), "name": body["name"], "fans_num": body["fans_num"], "gender":"男","figure":"普通用户","birth":"2000-01-01","traits":"乐观、耐心、敢于冒险","hobbies":"唱、跳、rap、篮球","memory": {"type":"阅读","content":"欢迎来到微博！随时随地，发现新鲜事~"}}]
#         data = pd.DataFrame(body)
#         data.to_csv("UserLoad.csv", mode="a", header=False, index=False)
#         return JSONResponse(content={"state":True,"id": str(len(df))})
#     except Exception as e:
#         return JSONResponse(content={"state":False, "error": str(e)}, status_code = 400)

# @app.delete("/UserDelete/{userID}")
# async def read_UserDelete(userID:str):
#     try:
#         df = pd.read_csv("UserLoad.csv")
#         df["userID"] = df["userID"].astype(str)
#         df = df[df["userID"] != userID]
#         df.to_csv("UserLoad.csv", mode="w", header=True, index=False)
#         return JSONResponse(content={"state":True})
#     except Exception as e:
#         return JSONResponse(content={"state":False, "error": str(e)}, status_code=400)