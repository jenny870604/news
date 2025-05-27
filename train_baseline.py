import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ======== 1. 讀取資料 ===========
print("🔄 載入資料中...")
df = pd.read_csv("data/tokenized_data.csv")  # 載入已斷詞好的新聞資料，欄位包含 text 和 label

# ======== 2. Label 編碼 ===========
# 將類別文字轉換為整數 id
label2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)  # 加入 label_id 欄位

# ======== 3. 過濾低樣本類別 ===========
# 只保留出現次數大於等於 2 的類別，避免過少資料導致訓練不穩
label_counts = df["label_id"].value_counts()
df_filtered = df[df["label_id"].isin(label_counts[label_counts >= 2].index)]

# ======== 4. 切分資料集 ===========
# 分成訓練集和測試集，並依照 label_id 分層取樣
train_df, test_df = train_test_split(
    df_filtered,
    test_size=0.2,
    stratify=df_filtered["label_id"],
    random_state=42
)

# ======== 5. 轉為 HuggingFace Dataset ===========
# 轉換成 Hugging Face 的 Dataset 格式，並將 label_id 欄位重新命名為 labels
train_dataset = Dataset.from_pandas(train_df.rename(columns={"label_id": "labels"})[["text", "labels"]])
test_dataset = Dataset.from_pandas(test_df.rename(columns={"label_id": "labels"})[["text", "labels"]])

# ======== 6. Tokenizer 處理 ===========
# 指定使用的模型名稱
# model_name = "uer/roberta-base-finetuned-chinanews-chinese"
# model_name = "hfl/chinese-roberta-wwm-ext"
model_name= "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 載入 tokenizer

# 定義 tokenizer 函式：加上 padding、truncation 並限制最大長度為 512
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

# 套用 tokenizer 到訓練和測試資料集
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# 移除原始 text 欄位，並設定格式為 torch tensor
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# ======== 7. 模型準備 ===========
# 載入預訓練的 BERT 模型，並設定分類類別數量
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    ignore_mismatched_sizes=True  # 若預訓練模型輸出維度不同則強制忽略
)

# ======== 8. 訓練設定 ===========
training_args = TrainingArguments(
    output_dir="./bert_results",              # 訓練結果輸出資料夾
    num_train_epochs=3,                       # 訓練 epoch 數
    per_device_train_batch_size=8,            # 每個裝置的訓練 batch 大小
    per_device_eval_batch_size=8,             # 每個裝置的驗證 batch 大小
    learning_rate=2e-5,                       # 學習率
    weight_decay=0.01,                        # 權重衰減
    logging_dir="./bert_logs",                # 日誌資料夾
    logging_steps=10,                         # 每 10 步驟紀錄一次
)

# ======== 9. 評估指標 ===========
accuracy = evaluate.load("accuracy")          # 載入 accuracy 指標
f1 = evaluate.load("f1")                      # 載入 F1-score 指標

# 定義評估函式：回傳 accuracy 與 macro-F1 分數
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

# ======== 10. Trainer 訓練 ===========
# 使用 Hugging Face 的 Trainer API 進行訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("🚀 開始訓練...")
trainer.train()  # 開始訓練模型

# ======== 11. 模型儲存 ===========
# 儲存訓練好的模型和 tokenizer
trainer.save_model("bert_models/roberta-news")
tokenizer.save_pretrained("bert_models/roberta-news")
