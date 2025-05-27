import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import fontManager
import matplotlib as mlp

# ===== 0. 設定中文字型（避免圖表中文字亂碼） =====
fontManager.addfont("ChineseFont.ttf")  # 請改為你系統上的中文字型路徑
mlp.rc("font", family="ChineseFont")    # 設定全域字型

# ===== 1. 載入資料與處理 =====
df = pd.read_csv("data/tokenized_data.csv")  # 載入之前預處理過的資料
label2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)  # 將文字標籤轉為數字

# 過濾樣本太少的類別（避免影響模型評估）
label_counts = df["label_id"].value_counts()
df_filtered = df[df["label_id"].isin(label_counts[label_counts >= 2].index)]

# 切出與訓練時一致的 test set
from sklearn.model_selection import train_test_split
_, test_df = train_test_split(
    df_filtered,
    test_size=0.2,
    stratify=df_filtered["label_id"],
    random_state=42
)

# ===== 2. 載入 tokenizer 與訓練好的模型 =====
model_path = "bert_models/roberta-news"  # 儲存模型的資料夾
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# ===== 3. 將 test 資料轉為 Hugging Face Dataset 並做 tokenization =====
test_dataset = Dataset.from_pandas(test_df.rename(columns={"label_id": "labels"})[["text", "labels"]])

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

test_dataset = test_dataset.map(tokenize, batched=True)        # 批次斷詞
test_dataset = test_dataset.remove_columns(["text"])           # 移除 text 欄位
test_dataset.set_format("torch")                               # 轉為 PyTorch 格式

# ===== 4. 執行預測 =====
from transformers import Trainer

trainer = Trainer(model=model, tokenizer=tokenizer)
pred_output = trainer.predict(test_dataset)  # 預測輸出：logits、標籤等

# 從 logits 取出預測類別
pred_labels = np.argmax(pred_output.predictions, axis=1)
true_labels = pred_output.label_ids

# ===== 5. 輸出分類報告 =====
# 組合實際與預測標籤出現過的類別
used_label_ids = sorted(set(true_labels) | set(pred_labels))
target_names = [id2label[i] for i in used_label_ids]  # 對應的文字標籤

# 印出分類報告（precision、recall、f1-score 等）
print("\n📋 分類報告：")
print(classification_report(true_labels, pred_labels, labels=used_label_ids, target_names=target_names))

# 儲存成 CSV 檔
report = classification_report(true_labels, pred_labels, labels=used_label_ids, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).T
report_df.to_csv("bert_classification_report.csv", encoding="utf-8-sig")

# ===== 6. 混淆矩陣視覺化 =====
cm = confusion_matrix(true_labels, pred_labels, labels=used_label_ids)  # 計算混淆矩陣

# 繪圖設定
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel("預測類別")
plt.ylabel("實際類別")
plt.title("分類混淆矩陣")
plt.tight_layout()
plt.savefig("bert_confusion_matrix.png", dpi=300)  # 儲存成高解析圖檔
plt.show()  # 顯示圖表
