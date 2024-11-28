# arXiv-Classifier-and-Finder
## 实验准备与步骤

### 1. 实验准备

#### 1.1 安装需要的包

```bash
pip install kaggle
pip install huggingface_hub git-lfs
pip install pandas numpy nltk
pip install sentence_transformers
pip install keras tensorflow
pip install streamlit
pip install faiss-cpu
# 或者选择 GPU 版本
pip install faiss-gpu
```

#### 1.2 获取数据集

##### 1.2.1 运行爬虫程序

1. 运行以下命令以执行爬虫程序：

   ```bash
   python3 arXiv_scraper.py
   ```

2. 根据需要修改爬虫程序中的搜索网址以获取不同的数据集。

##### 1.2.2 直接从网上获取爬取好的论文数据

1. 从 Kaggle 获取数据集：

   ```bash
   kaggle datasets download Cornell-University/arxiv
   unzip arxiv.zip
   ```

#### 1.3 下载模型

1. 安装 Git LFS：

   ```bash
   pip install git-lfs
   ```

2. 下载所需模型到本地（可以选择每次运行时在线加载）：

   ```bash
   git clone https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2
   git clone https://hf-mirror.com/cross-encoder/ms-marco-MiniLM-L-6-v2
   ```

3. 将模型保存在 `models` 文件夹中，并根据实际路径修改代码。

---

### 2. 数据处理

运行以下命令处理数据：

```bash
python3 arxiv_process.py
```

处理成功后将生成以下两个文件：

- `processed_data.json`：处理后的文本数据
- `compressed_array.npz`：生成的向量数据

---

### 3. 分类模型训练

#### 3.1 训练 MLP 模型

运行以下命令以训练 MLP 模型：

```bash
python3 arxiv_classifier_MLP.py
```

运行成功后，将生成训练结果，并保存模型为 `baseline.h5`。

#### 3.2 训练 BERT 模型

运行以下命令以训练 BERT 模型：

```bash
python3 arxiv_classifier_BERT.py
```

运行成功后，将生成训练结果，并保存以下文件：

- `arxiv_bert.h5`：训练好的 BERT 模型
- `mlb.pkl`：保存的 `MultiLabelBinarizer` 对象
- `vectorizer.pkl`：保存的 TF-IDF 向量化器

---

### 4. Web端可视化

运行以下命令以启动可视化 Web 应用：

```bash
streamlit run arxiv_project_webview.py
```

该程序将调用数据处理与训练模型生成的文件，并在浏览器中运行。根据实际文件结构修改代码中的路径。
