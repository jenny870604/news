from datasets import load_dataset
from ckip_transformers.nlp import CkipWordSegmenter
import pandas as pd
import os

# ======== 1. 載入資料 ===========
print("🔄 載入資料中...")
# 從 Hugging Face 載入 zh-tw-articles-6k 資料集的訓練集
dataset = load_dataset("AWeirdDev/zh-tw-articles-6k")["train"]

# 將 title 與 content 合併為一段文字，並取得標籤
texts = [item["title"] + "。" + item["content"] for item in dataset]
labels = dataset["tag"]  # 取得對應的新聞分類標籤

# ======== 2. 使用 CKIP 斷詞 ===========
# 初始化 CKIP 斷詞工具，device=-1 表示使用 CPU，如有 GPU 可改為 device=0
ws_driver = CkipWordSegmenter(device=-1)

tokenized_texts = []
batch_size = 32  # 每次處理 32 筆，以避免記憶體負擔

# 分批斷詞處理
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    tokens = ws_driver(batch)  # 斷詞
    tokenized_texts.extend([" ".join(tok) for tok in tokens])  # 合併詞為字串

# ======== 3. 儲存處理結果為 CSV ===========
# 將斷詞結果與標籤轉為 DataFrame
df = pd.DataFrame({"text": tokenized_texts, "label": labels})

# 建立資料夾（若不存在）
os.makedirs("data", exist_ok=True)

# 將結果儲存為 CSV 檔案，使用 UTF-8 with BOM 編碼以支援 Excel
df.to_csv("data/tokenized_data.csv", index=False, encoding="utf-8-sig")
print("✅ 已儲存斷詞後資料到 data/tokenized_data.csv")