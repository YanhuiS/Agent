import pandas as pd
from set import First_Name, Last_Name, Status, Trait, Interest
from distribution import age_distribution, init_atti_distribution, draw_distribution
import random
import numpy as np
from tqdm import tqdm

data = [] # 待存储信息
agent_number = 10000

# age是截断正态分布的年龄值，a1、b1是截断正态分布的标准化边界
age, a1, b1 = age_distribution(mean=30, sigma=20, lower=18, upper=np.inf, sample_size= agent_number)
# init_atti是年龄值，a2、b2是标准化边界
init_atti, a2, b2 = init_atti_distribution(mean=0,sigma=0.3,lower=-1,upper=1,sample_size= agent_number)
# 年龄的均值和标准差
mean1 = 30
sigma1 = 20
# 初始态度的均值和标准差
mean2 = 0
sigma2 = 0.5
draw_distribution(age, mean1, sigma1, a1, b1, init_atti, mean2, sigma2, a2, b2)
# draw_distribution(samples1, mean1, sigma1, a1, b1, samples2, mean2, sigma2, a2, b2)

for id in tqdm(range(0,agent_number),desc='processing agents'):
    
    first_name = random.choice(First_Name) 
    last_name = random.choice(Last_Name)
    new_name = f"{first_name} {last_name}"
    
    new_gender = random.choice(["male","female"])

    new_racist = random.choice(["white","black","asian","hispanic","other"])

    new_edu = random.choice(["high school","college","postgraduate"])

    new_economic = random.choice(["low","middle","high"])
    
    s_type = random.choice( list(Status.keys()) ) # Status's随机抽取3个键
    if age[id] > 30 and s_type == "Education" :
        new_status = random.choice( [status for status in Status[s_type] if status != "Student"] )
    else:
        new_status = random.choice( Status[s_type] )


    t_class = random.sample([i for i in range(5)], 3) # Trait随机抽取3个类别的下标（共5种类别，对应5个字典）
    # print(f"t_class: {t_class}")
    t_type = [random.choice(list(Trait[i].keys())) for i in t_class]# 每个类别中随机抽取正面或负面名词
    # print(f"t_type: {t_type}")
    new_traits = [random.choice(Trait[value][t_type[index]]) for index, value in enumerate(t_class)] # 每个名词选取1个形容词
    # print(f"new_traits: {new_traits}")
    new_interest = [random.choice(Interest[value][t_type[index]]) for index, value in enumerate(t_class)] # 注意，interest得和trait_type对齐
    # print(f"new_interest: {new_interest}")
    new_traits_string = ';'.join(new_traits)
    new_interest_string = ';'.join(new_interest)

    new_role_description = f"{new_name} is a {age[id]}-year-old {new_gender} {new_status} who is {new_traits[0]}, {new_traits[1]} and {new_traits[2]}, and has interests of {new_interest[0]}, {new_interest[1]} and {new_interest[2]}."

    new_agent = {
        'id':id,
        'name': new_name,
        'role_description': new_role_description,
        'init_att': init_atti[id],
        'gender': new_gender,
        'age': age[id],
        'status': new_status,
        'traits': new_traits_string,
        'interest': new_interest_string,
        'racist': new_racist,
        'education': new_edu,
        'economic': new_economic
    }

    data.append(new_agent)

# 将数据转换为pandas DataFrame
df = pd.DataFrame(data)
# 指定CSV文件名
filename = f'{agent_number}profile.csv'
# 写入CSV文件
df.to_csv(filename, index=False)

print(f'Data has been written to {filename}')