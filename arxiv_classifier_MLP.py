import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential


df = pd.read_json("processed_data.json", orient="records", lines=True)

df = df[:10000]
print(1)
mlb_df = MultiLabelBinarizer()
y_df_binarized = mlb_df.fit_transform(df['categories'])

vectorizer = TfidfVectorizer(max_features=1000)
df_vec = vectorizer.fit_transform(df['text'])

X_train, X_valid, y_train, y_valid = train_test_split(
    df_vec, y_df_binarized, 
    test_size=0.2, random_state=42
)
print(2)

def create_mlp():
    mlp = Sequential()
    mlp.add(Dense(256, activation='relu'))
    mlp.add(Dense(len(mlb_df.classes_), activation='sigmoid'))
    
    return mlp
mlp = create_mlp()
L_RATE = 1e-3
LOSS_FUNC = keras.losses.BinaryCrossentropy(from_logits=False)
OPTIMIZER = keras.optimizers.Adam(learning_rate=L_RATE)
EPOCHS = 15
BATCH_SIZE = 32
METRICS = [
    keras.metrics.Precision(name="Precision"),
    tfa.metrics.F1Score(
        name="F1-Score",
        num_classes=len(mlb_df.classes_),
        # Since the dataset is imbalanced, I'll use micro averaging.
        average="micro",
        threshold=0.5
    ),
]
def make_prediction(
    model: Sequential, X: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    y_pred = model.predict(X)
    return (y_pred > threshold).astype(int)
mlp.compile(
    loss=LOSS_FUNC,
    optimizer=OPTIMIZER,
    metrics=METRICS,
)
X_train.sort_indices()
X_valid.sort_indices()
history = mlp.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

mlp.save_weights("baseline.h5")
print("Model saved!")
# 训练模型并保存训练历史
plt.figure(figsize=(12, 5))
# 绘制损失曲线
plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss.png", dpi=300)
plt.close()

# 绘制指标曲线（以F1-Score为例）
plt.figure(figsize=(12, 5))
plt.subplot(1, 1, 1)
plt.plot(history.history['F1-Score'], label='Training F1-Score')
plt.plot(history.history['val_F1-Score'], label='Validation F1-Score')
plt.title('F1-Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.legend()
plt.savefig("F1.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 5))

plt.subplot(1, 1, 1)
plt.plot(history.history['Precision'], label='Training Precision')
plt.plot(history.history['val_Precision'], label='Validation Precision')
plt.title('Precision Curve')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.savefig("Precision.png", dpi=300)




