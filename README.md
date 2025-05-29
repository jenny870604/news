**📰 中文新聞分類專題**

本專案利用多種 BERT 系列模型進行中文新聞分類，支援模型訓練、預測、錯誤分析與互動式展示。

**📌 專案目的**

- 探討多種中文預訓練模型（如 BERT、RoBERTa）在新聞分類任務中的表現差異。
- 建立分類錯誤分析與可視化流程，提升模型應用價值。
- 提供簡單的 Streamlit Demo 展示模型推論能力。

**🔧 安裝套件**
pip install -r requirements.txt

**📊 前處理流程**
python preprocess.py

- 使用 CKIP 進行繁體中文斷詞
- 合併標題與內文
- 儲存為 tokenized_data.csv

**🌐 Streamlit Demo**
streamlit run app.py