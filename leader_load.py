import pandas as pd

def read_list():
    # 尝试读取Excel文件
    try:
        # 使用pandas读取Excel文件
        df = pd.read_excel('./test_influence.xlsx')
        
        # 检查所需的列是否存在
        required_columns = ['name', 'uid','favourites_count', 'followers_count',  'statuses_count','listed_count', 'content', 'replycount', 'retweetcount', 'favoritecount','gender','media','createdb','gender','avatar']
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
    'statuses_count': 'leader_repost_count',
    'replycount': 'leader_comment_count',
    'avatar': 'leader_pic',
    'createdb': 'create_time',
    }
    result = result.rename(columns=column_mapping)
    
    return result



result = read_list()
result1=result[['leader','leader_id']]
df_list = result1.to_dict(orient='records')

# print(df_list)
# print(result.dtypes)
# print(result[['leader','leader_id']])

# print(result.dtypes)
specific_id = 3255278642
specific_row = result.loc[result['leader_id'] == specific_id]
required_columns = ['leader', 'leader_id','leader_fan_count', 'leader_follower_count',  'leader_post_count','leader_liked_count', 'leader_repost_count', 'leader_comment_count', 'leader_pic']
result_to_transmit = specific_row[required_columns].to_dict(orient='records')
# print(result_to_transmit)