import pandas as pd
from textblob import TextBlob
def label_stance(content, target):
    # 这里假设了一个简单的例子，实际代码会根据环境和目标来判断
    if "oppose" in content.lower():
        return "Oppose"
    return "Support"  # 默认认为是支持的立场

# att2score函数定义
def att2score(content, target):
    # 使用TextBlob计算情感分数
    blob = TextBlob(content)
    score = blob.sentiment.polarity  # 获取情感极性分数

    # 根据立场调整符号
    stance = label_stance(content, target)
    if stance in ['Oppose']:
        sign = -1
    else:
        sign = 1

    # 返回符号调整后的分数和立场
    return sign * abs(score), stance

# 加载CSV文件
df = pd.read_csv('2024/2024_12/2024-12-01.csv')

# 假设target已经定义，你可以根据需要调整
target = 'some_target'  # 用你的目标填充

# 为每条文本计算态度分数和立场
df[['att', 'stance']] = df['text'].apply(lambda x: pd.Series(att2score(x, target)))

# 保存带有态度分数和立场的新CSV文件
df.to_csv('2024-12-01_att.csv', index=False)

print("CSV文件处理完毕，已保存结果")
