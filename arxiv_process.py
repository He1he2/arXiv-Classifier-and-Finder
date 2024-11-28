import json
import pandas as pd
import numpy as np
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# 下载 NLTK 需要的资源
nltk.download('punkt_tab')  # 下载分词器
nltk.download('stopwords')  # 下载停用词库

# 类别映射表，用于将缩写形式的类别转换为可读形式
category_map = {
    'acc-phys': 'Accelerator Physics',
    'adap-org': 'Not available',
    'q-bio': 'Not available',
    # 省略了部分映射内容，完整映射请查看上方代码
    'stat.ML': 'Machine Learning',
    'stat.TH': 'Statistics Theory'
}

# 加载数据集并筛选指定年份范围内的论文数据
def load_filtered_data(filepath, start_year, end_year):
    """
    加载指定年份范围内的论文数据。

    :param filepath: str, 文件路径
    :param start_year: int, 起始年份
    :param end_year: int, 结束年份
    :return: list, 筛选后的论文数据
    """
    filtered_data = []
    with open(filepath, 'r') as f:
        for line in f:
            paper = json.loads(line)  # 将 JSON 格式的字符串转换为字典
            update_date = paper.get("update_date")
            if update_date:  # 如果存在更新日期字段
                paper_date = datetime.strptime(update_date, "%Y-%m-%d")  # 转换为日期对象
                if start_year <= paper_date.year <= end_year:  # 判断是否在指定年份范围内
                    # 提取需要的字段
                    filtered_data.append({
                        "id": paper.get("id"),
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "categories": paper.get("categories"),
                    })
    return filtered_data


# 将论文类别从缩写形式转换为可读形式
def get_cat_text(x):
    """
    将类别缩写转换为完整名称。

    :param x: str, 类别缩写
    :return: str, 类别完整名称
    """
    cat_text = ''
    cat_list = x.split(' ')  # 按空格分割多个类别
    for i, item in enumerate(cat_list):
        cat_name = category_map[item]  # 使用映射表获取类别完整名称
        if cat_name != 'Not available':  # 跳过不可用类别
            if i == 0:
                cat_text = cat_name
            else:
                cat_text = cat_text + ', ' + cat_name  # 多个类别用逗号分隔
    return cat_text.strip()  # 去掉多余的空格


# 删除文本中的换行符
def clean_text(x):
    """
    删除文本中的换行符和首尾空格。

    :param x: str, 原始文本
    :return: str, 清理后的文本
    """
    new_text = x.replace("\n", " ")  # 替换换行符为空格
    return new_text.strip()  # 去掉首尾空格


# 预处理文本：小写化、去标点符号、去停用词
def preprocess_text(text):
    """
    预处理文本：包括小写化、去除标点符号、去除停用词。

    :param text: str, 原始文本
    :return: str, 预处理后的文本
    """
    text = text.lower()  # 转换为小写
    text = text.translate(str.maketrans('', '', string.punctuation))  # 去除标点符号
    words = word_tokenize(text)  # 分词
    words = [word for word in words if word not in stop_words]  # 去停用词
    return ' '.join(words)  # 重新组合为字符串


if __name__ == '__main__':
    # 文件路径
    file_path = "./arxiv-metadata-oai-snapshot.json"

    # 指定筛选的年份范围
    start_year = 2023
    end_year = 2024

    # 加载指定年份范围内的数据
    filtered_data = load_filtered_data(file_path, start_year, end_year)

    # 转换为 Pandas DataFrame 格式
    df = pd.DataFrame(filtered_data)

    print("截取23~24年数据，共{}篇论文".format(len(df)))

    # 转换类别为可读形式
    df['cat_text'] = df['categories'].apply(get_cat_text)

    print("转换特征")

    # 清理标题和摘要文本中的换行符
    df['title'] = df['title'].apply(clean_text)
    df['abstract'] = df['abstract'].apply(clean_text)

    print("删除换行符")

    # 加载 NLTK 停用词
    stop_words = set(stopwords.words('english'))

    # 对摘要文本进行预处理
    df['processed_abstract'] = df['abstract'].apply(preprocess_text)

    print("删除停用词")

    # 将标题和处理后的摘要合并为一个字段
    df['text'] = df['title'] + ' {title} ' + df['processed_abstract']

    print("合并标题和摘要")

    # 将处理后的数据保存为 JSON 文件
    df.to_json("processed_data.json", orient="records", lines=True, force_ascii=False)

    print("数据保存成功")

    # 提取文本、论文 ID 和类别信息
    chunk_list = list(df['text'])
    arxiv_id_list = list(df['id'])
    cat_list = list(df['cat_text'])

    # 加载 SentenceTransformer 模型
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("./models/all-MiniLM-L6-v2")
    print("模型加载成功")

    # 生成文本的嵌入表示
    embeddings = model.encode(chunk_list)

    # 将嵌入表示保存为压缩文件
    np.savez_compressed('compressed_array.npz', array_data=embeddings)

    print("embeddings 创建完成")
