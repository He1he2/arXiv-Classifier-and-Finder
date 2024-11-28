from lxml import html  # 用于解析HTML文档
import requests  # 用于发送HTTP请求
import re  # 正则表达式模块，用于匹配字符串
import math  # 数学模块，用于计算分页数
import csv  # 用于处理CSV文件
from bs4 import BeautifulSoup  # BeautifulSoup模块，用于HTML解析和数据提取
import time  # 用于设置爬取间隔，避免频繁请求导致被封禁

def get_total_results(url):
    """
    获取搜索结果的总数。

    参数:
        url (str): 需要爬取的搜索结果页面的URL。

    返回:
        int: 搜索结果总数。如果无法解析，则返回0。
    """
    # 发送HTTP GET请求获取页面内容
    response = requests.get(url)
    # 使用lxml的HTML解析器解析页面
    tree = html.fromstring(response.content)
    # 提取页面中的搜索结果统计字符串
    result_string = ''.join(tree.xpath('//*[@id="main-container"]/div[1]/div[1]/h1/text()')).strip()
    # 使用正则表达式提取结果总数
    match = re.search(r'of ([\d,]+) results', result_string)
    if match:
        # 将匹配的结果转换为整数
        total_results = int(match.group(1).replace(',', ''))
        return total_results
    else:
        # 如果没有找到匹配结果，打印提示信息并返回0
        print("没有找到匹配的数字。")
        return 0

def get_paper_info(url):
    """
    根据URL爬取一页的论文信息。

    参数:
        url (str): 需要爬取的页面URL。

    返回:
        list: 包含每篇论文信息的字典列表。
    """
    # 发送HTTP GET请求获取页面内容
    response = requests.get(url)
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    papers = []  # 用于存储论文信息的列表

    # 遍历页面中的每篇论文
    for article in soup.find_all('li', class_='arxiv-result'):
        # 提取论文的arXiv ID
        title_tag = article.find('p', class_='list-title')
        if title_tag:
            match = re.search(r'arXiv:(\d+\.\d+)', title_tag.text)
            if match:
                id = match.group(1)  # 提取 ID 部分
            else:
                id = 'No ID found'
        else:
            id = 'No ID found'

        # 提取论文标题
        title = article.find('p', class_='title').text.strip()

        # 提取作者信息
        authors_text = article.find('p', class_='authors').text.replace('Authors:', '').strip()
        authors = [author.strip() for author in authors_text.split(',')]

        # 提取摘要信息
        abstract = article.find('span', class_='abstract-full').text.strip()

        # 提取提交日期
        submitted = article.find('p', class_='is-size-7').text.strip()
        submission_date = submitted.split(';')[0].replace('Submitted', '').strip()

        # 提取PDF链接
        pdf_link_element = article.find('a', text='pdf')
        if pdf_link_element:
            pdf_link = pdf_link_element['href']
        else:
            pdf_link = 'No PDF link found'

        # 将提取的信息存入字典并添加到列表中
        papers.append({
            'id': id,
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'submission_date': submission_date,
            'pdf_link': pdf_link
        })
    
    return papers

def save_to_csv(papers, filename):
    """
    将爬取到的论文信息保存到CSV文件中。

    参数:
        papers (list): 包含每篇论文信息的字典列表。
        filename (str): 要保存的CSV文件名。
    """
    # 打开CSV文件进行写操作
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # 定义CSV的列名
        fieldnames = ['id', 'title', 'authors', 'abstract', 'submission_date', 'pdf_link']
        # 创建CSV写入器
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 写入表头
        writer.writeheader()
        # 写入每篇论文的信息
        for paper in papers:
            writer.writerow(paper)

# 主程序入口
base_url = "https://arxiv.org/search/?searchtype=all&query=attack&abstracts=show&size=200&order=-announced_date_first&date-date_type=submitted_date"

# 获取总结果数
total_results = get_total_results(base_url + "&start=0")
# 计算总页数（每页200条）
pages = math.ceil(total_results / 200)
all_papers = []  # 用于存储所有页的论文信息

# 遍历每页爬取数据
for page in range(2):  # 示例仅爬取前两页
    start = page * 200  # 计算当前页的起始记录数
    print(f"Crawling page {page+1}/{pages}, start={start}")
    page_url = base_url + f"&start={start}"  # 构造当前页的URL
    # 获取当前页的论文信息并添加到总列表中
    all_papers.extend(get_paper_info(page_url))
    time.sleep(3)  # 等待3秒以避免频繁请求导致封禁

# 保存所有爬取的论文信息到CSV文件
save_to_csv(all_papers, 'data.json')
print(f"完成！总共爬取到 {len(all_papers)} 条数据，已保存到文件中。")
