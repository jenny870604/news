import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report
import os
import json

# === 目錄設定 ===
MODEL_DIR = "./bert_models/roberta-news"
TEST_DATA_PATH = "./data/test_data.csv"
REPORT_DIR = "./report"
os.makedirs(REPORT_DIR, exist_ok=True)

# === 手動指定 label2id（必須與訓練時一致） ===
label2id = {
    "中央社": 0, "中港澳": 1, "公共政策": 2, "公民運動": 3, "品味生活": 4,
    "國內": 5, "國際": 6, "政治": 7, "科技": 8, "評論": 9, "調查": 10, "財經": 11, "風生活": 12
}
id2label = {v: k for k, v in label2id.items()}

# === 載入測試資料 ===
df_test = pd.read_csv(TEST_DATA_PATH)
df_test["label_id"] = df_test["label"].map(label2id)

# === Tokenizer 與 Model 載入 ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=len(label2id))

# === 建立 test dataset ===
test_dataset = Dataset.from_pandas(df_test[["text", "label_id"]])

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.rename_column("label_id", "labels")
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# === Trainer 建立與預測 ===
trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset)
pred_label = np.argmax(predictions.predictions, axis=1)
true_label = predictions.label_ids

# === 分類報告 ===
labels = sorted(id2label.keys())
target_names = [id2label[i] for i in labels]

report = classification_report(
    true_label,
    pred_label,
    labels=labels,
    target_names=target_names,
    output_dict=True
)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(f"{REPORT_DIR}/classification_report.csv", encoding="utf-8-sig")
print("✅ 分類報告已儲存至 classification_report.csv")

# === 錯誤樣本分析 ===
df_test = df_test.reset_index(drop=True)
df_test["true_label_id"] = true_label
df_test["pred_label_id"] = pred_label
df_test["true_label"] = df_test["true_label_id"].map(id2label)
df_test["pred_label"] = df_test["pred_label_id"].map(id2label)
wrong_df = df_test[df_test["true_label_id"] != df_test["pred_label_id"]]
wrong_df.to_csv(f"{REPORT_DIR}/wrong_predictions.csv", encoding="utf-8-sig", index=False)
print("⚠️ 錯誤預測已儲存至 wrong_predictions.csv")
