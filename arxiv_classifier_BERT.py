# 导入必要的库
import re
import ast
import time
import json
import pickle
import random
import inspect
import datetime as dt
import multiprocessing
from typing import Dict, Any, Generator, Tuple, Optional

# 常用科学计算和数据处理库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn 用于数据分割、特征提取和多标签处理
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

# TensorFlow 和 TensorFlow-Addons 用于深度学习模型构建
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Dense, Dropout
from keras.models import Sequential

# Transformers 库，用于加载预训练模型和分词器
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    TFAutoModel,
)

# 禁用部分优化标志，防止 XLA 的不兼容问题
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# 加载数据集
df = pd.read_json("third_data.json", orient="records", lines=True)

# 限制数据集规模到前 10000 条记录，以便于快速训练和测试
df = df[:10000]
print(1)

# 初始化 MultiLabelBinarizer，将类别标签转为多标签二值化格式
mlb = MultiLabelBinarizer()
y_df_binarized = mlb.fit_transform(df['categories'])

# 将 MultiLabelBinarizer 对象保存到文件，以便后续加载使用
with open("mlb.pkl", "wb") as mlb_file:
    pickle.dump(mlb, mlb_file)

# 初始化 TF-IDF 向量化器，用于将文本转化为向量表示
vectorizer = TfidfVectorizer(max_features=1000)  # 只保留 1000 个最高频特征
df_vec = vectorizer.fit_transform(df['text'])

# 将 TF-IDF 向量化器保存到文件中
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# 将数据集分为训练集和验证集，比例为 80%:20%
X_train, X_valid, y_train, y_valid = train_test_split(
    df_vec, y_df_binarized, 
    test_size=0.2, random_state=42  # 固定随机种子以保证结果可重复
)
print(2)

# 定义最大序列长度和使用的预训练模型 ID
MAX_SEQ_LEN = 200
MODEL_ID = "bert-base-uncased"

# 加载预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 定义评估指标，包括 Precision 和 F1-Score
METRICS = [
    tf.keras.metrics.Precision(name="Precision"),  # 精确率
    tfa.metrics.F1Score(  # F1 分数
        name="F1-Score",
        num_classes=len(mlb.classes_),  # 类别数
        average="micro",  # 微平均方式计算
        threshold=0.5  # 阈值
    ),
]

# 定义一个基于预训练模型 BERT 的自定义模型
class ArxivBert(tf.keras.Model):
    def __init__(self, base_model_id: str, num_labels: int):
        super().__init__()
        # 加载 BERT 预训练模型作为基础模型
        self._base = TFAutoModel.from_pretrained(base_model_id, from_pt=True)
        self._base.trainable = True  # 允许对 BERT 层进行微调
        
        # 添加额外的全连接层，用于多标签分类
        self._additional_layers = tf.keras.Sequential([
            Dropout(0.1),  # Dropout 防止过拟合
            Dense(512, activation="relu"),  # 隐藏层
            Dense(256, activation="relu"),  # 隐藏层
            Dense(num_labels, activation="sigmoid"),  # 输出层，激活函数为 Sigmoid
        ])
        
    def call(self, inputs):
        # 前向传播：先通过 BERT 提取特征，然后通过额外的层进行分类
        out = self._base(inputs)
        out = out["last_hidden_state"][:, 0, :]  # 提取 [CLS] token 的嵌入向量
        return self._additional_layers(out)

# 初始化自定义模型
arxiv_bert = ArxivBert(MODEL_ID, len(mlb.classes_))

# 对训练和验证集的文本进行分词和编码
train_encodings = tokenizer(
    df["text"][:8000].tolist(),  # 使用前 8000 条记录作为训练集
    truncation=True,  # 截断到最大长度
    padding="max_length",  # 填充到最大长度
    max_length=MAX_SEQ_LEN,  # 最大序列长度
    return_tensors="tf"  # 返回 TensorFlow 格式的张量
)

valid_encodings = tokenizer(
    df["text"][8000:].tolist(),  # 使用后 2000 条记录作为验证集
    truncation=True,
    padding="max_length",
    max_length=MAX_SEQ_LEN,
    return_tensors="tf"
)

# 构建模型输入：输入 ID 和注意力掩码
X_train_enc = {
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
}
X_valid_enc = {
    "input_ids": valid_encodings["input_ids"],
    "attention_mask": valid_encodings["attention_mask"],
}

# 将标签转化为 TensorFlow 张量
y_train_enc = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_valid_enc = tf.convert_to_tensor(y_valid, dtype=tf.float32)

# 定义训练的超参数
L_RATE = 2e-5  # 学习率
EPOCHS = 15  # 训练轮次
BATCH_SIZE = 32  # 每次训练的批量大小

# 初始化优化器和损失函数
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=L_RATE)
LOSS_FUNC = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# 编译模型，指定损失函数、优化器和评估指标
arxiv_bert.compile(
    loss=LOSS_FUNC,
    optimizer=OPTIMIZER,
    metrics=METRICS,
    jit_compile=True  # 启用加速编译
)

# 训练模型
history = arxiv_bert.fit(
    X_train_enc,  # 训练集输入
    y_train_enc,  # 训练集标签
    validation_data=(X_valid_enc, y_valid_enc),  # 验证集
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# 保存训练好的模型权重
arxiv_bert.save_weights("arxiv_bert.h5")
print("Model saved!")

# 绘制训练和验证的损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss2.png", dpi=300)
plt.close()

# 绘制训练和验证的 F1 分数曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 1, 1)
plt.plot(history.history['F1-Score'], label='Training F1-Score')
plt.plot(history.history['val_F1-Score'], label='Validation F1-Score')
plt.title('F1-Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.legend()
plt.savefig("F12.png", dpi=300)
plt.close()

# 绘制训练和验证的精确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 1, 1)
plt.plot(history.history['Precision'], label='Training Precision')
plt.plot(history.history['val_Precision'], label='Validation Precision')
plt.title('Precision Curve')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.savefig("Precision2.png", dpi=300)
