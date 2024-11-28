# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow_addons as tfa
import keras
from keras.layers import Dense
from keras.models import Sequential
from datasets import load_dataset

# 读取预处理后的数据集文件
df = pd.read_json("processed_data.json", orient="records", lines=True)

# 只选择前10000条数据进行训练，避免过大的数据导致内存溢出
df = df[:10000]
print(1)

# 初始化 MultiLabelBinarizer，用于将类别标签进行多标签二值化
mlb_df = MultiLabelBinarizer()
# 将类别标签（多标签格式）转化为二进制矩阵
y_df_binarized = mlb_df.fit_transform(df['categories'])

# 初始化 TfidfVectorizer，用于将文本数据转化为 TF-IDF 特征向量
vectorizer = TfidfVectorizer(max_features=1000)  # 限制最多1000个特征
# 将文本列数据进行向量化处理
df_vec = vectorizer.fit_transform(df['text'])

# 将数据集划分为训练集和验证集，比例为 80%:20%
X_train, X_valid, y_train, y_valid = train_test_split(
    df_vec, y_df_binarized,  # 输入特征和标签
    test_size=0.2,  # 验证集占 20%
    random_state=42  # 固定随机种子，确保每次分割结果相同
)
print(2)

# 创建一个多层感知机（MLP）模型的函数
def create_mlp():
    mlp = Sequential()
    # 添加一个全连接层，输出维度为256，激活函数为ReLU
    mlp.add(Dense(256, activation='relu'))
    # 输出层，节点数为类别数，激活函数为Sigmoid（用于多标签分类）
    mlp.add(Dense(len(mlb_df.classes_), activation='sigmoid'))
    return mlp

# 初始化多层感知机模型
mlp = create_mlp()

# 定义模型训练的超参数
L_RATE = 1e-3  # 学习率
LOSS_FUNC = keras.losses.BinaryCrossentropy(from_logits=False)  # 损失函数
OPTIMIZER = keras.optimizers.Adam(learning_rate=L_RATE)  # 优化器
EPOCHS = 15  # 训练轮次
BATCH_SIZE = 32  # 每次训练的批量大小

# 定义模型评估的指标，包括 Precision 和 F1-Score
METRICS = [
    keras.metrics.Precision(name="Precision"),  # 精确率
    tfa.metrics.F1Score(  # F1分数
        name="F1-Score",
        num_classes=len(mlb_df.classes_),  # 类别数量
        average="micro",  # 微平均方式计算 F1
        threshold=0.5  # 阈值
    ),
]

# 定义一个函数，用于对输入数据进行预测并应用阈值
def make_prediction(
    model: Sequential, X: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    # 模型预测，返回的是概率值
    y_pred = model.predict(X)
    # 根据阈值将概率值转化为二进制标签
    return (y_pred > threshold).astype(int)

# 编译模型，指定损失函数、优化器和评估指标
mlp.compile(
    loss=LOSS_FUNC,
    optimizer=OPTIMIZER,
    metrics=METRICS,
)

# 修复稀疏矩阵的索引问题（避免 Keras 报错）
X_train.sort_indices()
X_valid.sort_indices()

# 开始训练模型
history = mlp.fit(
    X_train,  # 训练集输入特征
    y_train,  # 训练集标签
    validation_data=(X_valid, y_valid),  # 验证集
    epochs=EPOCHS,  # 训练轮次
    batch_size=BATCH_SIZE  # 每次训练的批量大小
)

# 保存模型权重到文件中
mlp.save_weights("baseline.h5")
print("Model saved!")

# 绘制训练和验证的损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')  # 训练损失
plt.plot(history.history['val_loss'], label='Validation Loss')  # 验证损失
plt.title('Loss Curve')  # 图标题
plt.xlabel('Epochs')  # 横轴标签
plt.ylabel('Loss')  # 纵轴标签
plt.legend()  # 显示图例
plt.savefig("loss.png", dpi=300)  # 保存图片
plt.close()

# 绘制训练和验证的 F1 分数曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 1, 1)
plt.plot(history.history['F1-Score'], label='Training F1-Score')  # 训练 F1 分数
plt.plot(history.history['val_F1-Score'], label='Validation F1-Score')  # 验证 F1 分数
plt.title('F1-Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.legend()
plt.savefig("F1.png", dpi=300)
plt.close()

# 绘制训练和验证的精确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 1, 1)
plt.plot(history.history['Precision'], label='Training Precision')  # 训练精确率
plt.plot(history.history['val_Precision'], label='Validation Precision')  # 验证精确率
plt.title('Precision Curve')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.savefig("Precision.png", dpi=300)
