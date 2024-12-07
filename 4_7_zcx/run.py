# import random
# import pandas as pd

# # 加载原始 CSV 文件
# df = pd.read_csv('profile_10w.csv')

# # 为每一行数据添加新列
# df['likes'] = [random.randint(0, 1000) for _ in range(len(df))]
# df['retweets'] = [random.randint(0, 100) for _ in range(len(df))]
# df['comments'] = [random.randint(0, 200) for _ in range(len(df))]
# df['views'] = [random.randint(0, 10000) for _ in range(len(df))]
# df['collects'] = [random.randint(0, 100) for _ in range(len(df))]
# df['roles'] = 0
# df.loc[:9999, 'roles'] = [random.choice(["News media", "Opinion leaders"]) for _ in range(10000)]
# df['uid'] = [random.randint(10000, 20000) for _ in range(len(df))]

# # 将更新后的 DataFrame 保存为新的 CSV 文件
# df.to_csv('updated_data.csv', index=False, encoding='utf-8')

# print("数据已成功保存到 updated_data.csv")



import pandas as pd

# 读取 CSV 数据
df = pd.read_csv('newprofile_10w.csv')

# 定义职业和大类的映射
category_mapping = {
    "Arts": ["Artist", "Musician", "Actor", "Sculptor", "Photographer", "Graphic Designer", 
             "Dancer", "Writer", "Filmmaker", "Curator"],
    "Business": ["Entrepreneur", "Business Analyst", "Financial Manager", "Marketing Specialist", 
                 "Operations Manager", "Sales Executive", "Consultant", "Project Manager", 
                 "Human Resources Manager", "Accountant"],
    "Communications": ["Public Relations Specialist", "Journalist", "Content Writer", 
                       "Social Media Manager", "Editor", "Copywriter", "Communications Director", 
                       "Speechwriter", "Broadcast Journalist", "Advertising Executive"],
    "Education": ["Student", "Professor", "Educational Administrator", "Curriculum Developer", 
                  "School Counselor", "Librarian", "Special Education Teacher", "Instructional Designer", 
                  "Tutor", "Education Policy Analyst"],
    "Healthcare": ["Physician", "Nurse", "Pharmacist", "Physical Therapist", "Occupational Therapist", 
                   "Radiologic Technologist", "Medical Researcher", "Health Educator", 
                   "Dental Hygienist", "Physician Assistant"],
    "Hospitality": ["Hotel Manager", "Chef", "Event Planner", "Front Desk Associate", "Bartender", 
                    "Travel Agent", "Restaurant Manager", "Tour Guide", "Concierge", "Housekeeping Supervisor"],
    "Information Technology": ["Software Developer", "System Administrator", "Data Analyst", "IT Support Specialist", 
                               "Network Engineer", "Cybersecurity Analyst", "Web Developer", "Database Administrator", 
                               "Cloud Engineer", "UX/UI Designer"],
    "Law Enforcement": ["Police Officer", "Detective", "Criminal Investigator", "Forensic Analyst", 
                        "Correctional Officer", "Security Consultant", "Crime Scene Technician", 
                        "Community Liaison Officer", "Federal Agent", "Dispatcher"],
    "Sales and Marketing": ["Sales Manager", "Brand Strategist", "Market Research Analyst", 
                            "Digital Marketing Specialist", "Account Executive", "SEO Specialist", 
                            "Product Manager", "Customer Service Representative", "Promotions Coordinator", 
                            "Trade Show Manager"],
    "Science": ["Research Scientist", "Lab Technician", "Environmental Scientist", "Biologist", 
                "Chemist", "Physicist", "Data Scientist", "Astronomer", "Geologist", "Marine Biologist"],
    "Transportation": ["Truck Driver", "Airline Pilot", "Logistics Coordinator", "Transportation Planner", 
                       "Rail Operator", "Shipping Manager", "Fleet Manager", "Traffic Engineer", 
                       "Air Traffic Controller", "Delivery Driver"]
}

# 为每个大类分配颜色
category_colors = {
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
def get_category_color(status):
    for category, jobs in category_mapping.items():
        for job in jobs:
            if job in status:  # 判断status是否包含该职业
                return category_colors[category]
    return "unknown"  # 如果没有找到对应的职业，则返回"unknown"

# 将颜色分配到新的列
df['category_color'] = df['status'].apply(get_category_color)

# 保存更新后的 DataFrame 为新的 CSV 文件
df.to_csv('updated_data_with_colors.csv', index=False, encoding='utf-8')

print("数据已成功保存到 updated_data_with_colors.csv")


# import pandas as pd

# # 读取前 10000 行数据
# df = pd.read_csv('1newprofile_10w.csv', nrows=1000)

# # 保存到新的 CSV 文件
# df.to_csv('data_10000.csv', index=False)