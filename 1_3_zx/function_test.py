import os
import json
import re
import pandas as pd
from datetime import datetime, timedelta

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

def read_Statistics(observe_time: str):
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
    if observe_time == "day": deltatime = 1
    elif observe_time == "week": deltatime = 6
    else: deltatime = 29
    start_time = end_time - timedelta(days=deltatime)

    total = cal_total(df, end_time, start_time)
    print("total:", total)
    ring_growth = cal_ring_growth(df, deltatime, end_time, start_time)
    print("ring_growth:", ring_growth)
    average =cal_average(df, end_time, start_time) 
    print("average:", average)
    bias =cal_bias(df, end_time, start_time)
    print("bias:", bias)
    var =cal_var(df, end_time, start_time)
    print("var:", var)
    neg_total=cal_NegTotal(df, end_time, start_time)
    print("neg_total", neg_total)
    non_neg_total =cal_NNegTotal(df, end_time, start_time)
    print("non_neg_total", non_neg_total)
    leader_num=cal_leader_num(df, end_time, start_time) 
    print("leader_num:", leader_num)
    fluc=cal_fluc(df, deltatime, end_time, start_time)
    print("fluc:", fluc)
    spread_rate=cal_spread_rate(df, deltatime, end_time, start_time)
    print("spread_rate:", spread_rate)
    convert_rate=cal_convert_rate(df, deltatime, end_time, start_time)
    print("convert_rate:", convert_rate)
    grow_rate=cal_grow_rate(df, deltatime, end_time, start_time)
    print("grow_rate:", grow_rate)

read_Statistics("month")
 

