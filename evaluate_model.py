import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.font_manager import fontManager
import matplotlib as mlp

# ===== 0. 設定中文字型（避免圖表中文字亂碼） =====
fontManager.addfont("ChineseFont.ttf")  # 請改為你系統上的中文字型路徑
mlp.rc("font", family="ChineseFont")    # 設定全域字型

# === 1. 讀取三個模型的 F1-score 分類報告 ===
roberta_df = pd.read_csv("results/roberta_classification_report.csv")
chinese_df = pd.read_csv("results/chinese_classification_report.csv")
bert_df = pd.read_csv("results/bert_classification_report.csv")

# === 2. 整理欄位名稱 ===
roberta_df = roberta_df.rename(columns={"Unnamed: 0": "label", "f1-score": "roberta_f1"})
chinese_df = chinese_df.rename(columns={"Unnamed: 0": "label", "f1-score": "chinese_f1"})
bert_df = bert_df.rename(columns={"Unnamed: 0": "label", "f1-score": "bert_f1"})

# === 3. 合併三份資料 ===
combined_df = roberta_df[["label", "roberta_f1"]].merge(
    chinese_df[["label", "chinese_f1"]], on="label"
).merge(
    bert_df[["label", "bert_f1"]], on="label"
)

# === 4. 輸出合併結果 ===
combined_df.to_csv("results/combined_f1_scores.csv", index=False, encoding="utf-8-sig")

#刪除中港澳,公共政策
# combined_df = combined_df[~combined_df["label"].isin(["中港澳", "公共政策"])].reset_index(drop=True)

# === 5. 畫出三模型 F1-score 比較圖 ===
plt.figure(figsize=(12, 8))
x = range(len(combined_df))
bar_width = 0.25

plt.bar([i - bar_width for i in x], combined_df["roberta_f1"], width=bar_width, label="Roberta")
plt.bar(x, combined_df["chinese_f1"], width=bar_width, label="Chinese-RoBERTa")
plt.bar([i + bar_width for i in x], combined_df["bert_f1"], width=bar_width, label="BERT")

plt.xticks(ticks=x, labels=combined_df["label"], rotation=45, ha="right")
plt.ylabel("F1-score")
plt.title("三種模型 F1-score 分類表現比較")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.savefig("results/f1_score_comparison-2.png", dpi=300)
plt.show()
