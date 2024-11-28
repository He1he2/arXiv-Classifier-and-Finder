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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Dense, Dropout
from keras.models import Sequential
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    TFAutoModel,
)
from keras.callbacks import Callback
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Load data
df = pd.read_json("third_data.json", orient="records", lines=True)
df = df[:10000]  # Limit dataset for testing
print(1)

# Multi-label encoding
mlb = MultiLabelBinarizer()
y_df_binarized = mlb.fit_transform(df['categories'])
with open("mlb.pkl", "wb") as mlb_file:
    pickle.dump(mlb, mlb_file)
    
# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)
df_vec = vectorizer.fit_transform(df['text'])
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
    
# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(
    df_vec, y_df_binarized, 
    test_size=0.2, random_state=42
)
print(2)

MAX_SEQ_LEN = 200
MODEL_ID = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
METRICS = [
    tf.keras.metrics.Precision(name="Precision"),
    tfa.metrics.F1Score(
        name="F1-Score",
        num_classes=len(mlb.classes_),
        average="micro",
        threshold=0.5,
    ),
]
class ArxivBert(tf.keras.Model):
    def __init__(self, base_model_id: str, num_labels: int):
        super().__init__()
        self._base = TFAutoModel.from_pretrained(base_model_id, from_pt=True)
        self._base.trainable = True
        
        self._additional_layers = tf.keras.Sequential([
            Dropout(0.1),
            Dense(512, activation="relu"),
            Dense(256, activation="relu"),
            Dense(num_labels, activation="sigmoid"),
        ])
        
    def call(self, inputs):
        out = self._base(inputs)
        out = out["last_hidden_state"][:, 0, :]
        return self._additional_layers(out)
arxiv_bert = ArxivBert(MODEL_ID, len(mlb.classes_))
train_encodings = tokenizer(
    df["text"][:8000].tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_SEQ_LEN,
    return_tensors="tf"
)

valid_encodings = tokenizer(
    df["text"][8000:].tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_SEQ_LEN,
    return_tensors="tf"
)
X_train_enc = {
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
}
X_valid_enc = {
    "input_ids": valid_encodings["input_ids"],
    "attention_mask": valid_encodings["attention_mask"],
}


y_train_enc = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_valid_enc = tf.convert_to_tensor(y_valid, dtype=tf.float32)
L_RATE = 2e-5
EPOCHS = 15
BATCH_SIZE = 32
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=L_RATE)
LOSS_FUNC = tf.keras.losses.BinaryCrossentropy(from_logits=False)
arxiv_bert.compile(
    loss=LOSS_FUNC,
    optimizer=OPTIMIZER,
    metrics=METRICS,
    jit_compile=True
)
history = arxiv_bert.fit(
    X_train_enc,
    y_train_enc,
    validation_data=(X_valid_enc, y_valid_enc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# Save the model weights
arxiv_bert.save_weights("arxiv_bert.h5")
print("Model saved!")

plt.figure(figsize=(12, 5))
# 绘制损失曲线
plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss2.png", dpi=300)
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
plt.savefig("F12.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 5))

plt.subplot(1, 1, 1)
plt.plot(history.history['Precision'], label='Training Precision')
plt.plot(history.history['val_Precision'], label='Validation Precision')
plt.title('Precision Curve')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.savefig("Precision2.png", dpi=300)
