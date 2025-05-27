# 原始的資料分類
# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib.font_manager import fontManager
# import matplotlib as mlp

# # ===== 0. 設定中文字型（避免圖表中文字亂碼） =====
# fontManager.addfont("ChineseFont.ttf")  # 請改為你系統上的中文字型路徑
# mlp.rc("font", family="ChineseFont")    # 設定全域字型

# # 讀取 CSV
# df = pd.read_csv("data/tokenized_data.csv")  # 確保路徑正確

# # 顯示每個類別的數量
# label_counts = df["label"].value_counts()

# print("🔢 類別數量統計：")
# print(label_counts)

# label_counts.plot(kind='bar', figsize=(10, 5), title="各類別新聞數量")
# plt.xlabel("類別")
# plt.ylabel("數量")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# label 的種類
# import pandas as pd

# # ===== 1. 讀取資料 =====
# df = pd.read_csv("data/tokenized_data.csv")

# # ===== 2. 建立 label2id 和 id2label =====
# label2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
# id2label = {v: k for k, v in label2id.items()}

# # ===== 3. 顯示 id2label 映射 =====
# print("🆔 id2label 映射：")
# for id, label in id2label.items():
#     print(f"{id}: {label}")

# # ===== 4. 儲存成 CSV =====
# id2label_df = pd.DataFrame(list(id2label.items()), columns=["label_id", "label"])
# id2label_df.to_csv("data/id2label_mapping.csv", index=False, encoding="utf-8-sig")
# print("✅ 映射已儲存為 id2label_mapping.csv")

# 切割資料集
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/tokenized_data.csv")

# 找出每個類別的數量
label_counts = df["label"].value_counts()

# 只保留樣本數 >= 2 的類別
valid_labels = label_counts[label_counts >= 2].index
df_filtered = df[df["label"].isin(valid_labels)]

# stratify 切分
train_df, test_df = train_test_split(
    df_filtered, test_size=0.2, stratify=df_filtered["label"], random_state=42
)

test_df.to_csv("data/test_data.csv", index=False, encoding="utf-8-sig")
print("✅ 測試資料已儲存（已過濾低樣本類別）")

