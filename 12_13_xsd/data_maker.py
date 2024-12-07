import pandas as pd
import random
from faker import Faker
import datetime
import time

# 使用 Faker 来生成随机用户信息和评论内容
fake = Faker()

# 配置生成数据的数量
data_size = 100

# 准备评论内容的样本列表，模拟大选相关评论
comments_samples = {
    "positive": [
        "I fully support the current candidate, their policies are exactly what our country needs!",
        "I'm so excited to vote this year, it's important for everyone to have their voice heard.",
        "I believe in change, and this election is the perfect opportunity to make it happen.",
        "Great leadership is what we need, and this candidate can provide that!",
        "The candidate's stance on healthcare is exactly what this country needs."
    ],
    "negative": [
        "The election process seems flawed, we need better transparency to trust the results.",
        "This candidate has no real plan, just empty promises. We deserve better leadership.",
        "The debate was disappointing, none of the candidates addressed the real issues we're facing.",
        "The media is clearly biased towards one candidate, it's unfair to the voters.",
        "This candidate is completely unfit for office, their policies are a disaster waiting to happen."
    ],
    "neutral": [
        "No matter who wins, we must stay united as a country. Division will only harm us.",
        "I'm tired of all the negative campaigning, why can't they focus on real solutions?",
        "This election is the most important one in our lifetime, make sure to vote!",
        "It's important that everyone participates in the election, regardless of their political views.",
        "The candidates have some good points, but also some flaws. We need more balanced discussions."
    ]
}

# 准备地点样本列表（美国的所有州）
locations_samples = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", 
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", 
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", 
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", 
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

# 生成模拟数据
data = []
for _ in range(data_size):
    user_id = fake.uuid4()  # 随机生成用户 ID
    comment_time = int(fake.date_time_between(start_date='-1y', end_date='now').timestamp())  # 过去一年的随机时间，转换为时间戳
    sentiment = random.choice(["positive", "negative", "neutral"])  # 随机选择评论的情感类型
    comment_content = random.choice(comments_samples[sentiment])  # 根据情感类型选择评论内容
    comment_location = random.choice(locations_samples)  # 随机选择评论地点
    
    data.append([user_id, comment_time, comment_content, comment_location])

# 将数据放入 DataFrame
columns = ["user_id", "comment_time", "comment_content", "comment_location"]
df = pd.DataFrame(data, columns=columns)

# 打印生成的数据
print(df.head())

# 如果需要，可以将数据保存到 CSV 文件中
df.to_csv("simulated_social_media_data.csv", index=False)
