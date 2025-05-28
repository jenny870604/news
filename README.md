**📁 專案結構**
news/
├── bert_models/
│   └── roberta-news/             # bert 儲存訓練好的模型
├── bert_logs/                        # 訓練過程的日誌（loss, learning rate 變化等）
├── bert_result/                      # 訓練結果輸出資料夾
|── chinese_models/
│   └── roberta-news/             # chinese 儲存訓練好的模型
├── chinese_logs/                        # 訓練過程的日誌（loss, learning rate 變化等）
├── chinese_result/                      # 訓練結果輸出資料夾
├── data/
│   ├── tokenized_data.csv        # 預處理後的新聞文本與標籤
│   └── test_data.csv             # 測試資料集
├── report/
│   ├── classification_report.csv # 模型分類報告
│   └── wrong_predictions.csv     # 分類錯誤的樣本
├── result/                      # 各模型的預測結果、統計分析、可視化圖表
│── roberta_models/
│   └── roberta-news/             # roberta 儲存訓練好的模型
├── roberta_logs/                        # 訓練過程的日誌（loss, learning rate 變化等）
├── roberta_result/                      # 訓練結果輸出資料夾
├── app.py                       # Streamlit 互動式 Demo
├── train_baseline.py            # 模型訓練程式
├── evaluate_model.py                  # 合併三個模型預測與產出報告
├── evaluate_saved_model.py                  # 預測與產出報告
├── evaluate_report.py                  # 錯誤預測報告
├── preprocess.py                 # CKIP 中文斷詞與資料清洗
├── requirements.txt             # 相依套件清單
└── README.md                    # 專案說明文件


**🔧 安裝套件**
pip install -r requirements.txt

**📊 前處理流程**
python preprocess.py

-使用 CKIP 進行繁體中文斷詞

-合併標題與內文

-儲存為 tokenized_data.csv

**🏋️ 模型訓練**
python train_baseline.py

-使用 AutoModelForSequenceClassification（如 RoBERTa）建立分類模型

-設定超參數與訓練策略

-儲存訓練模型至 bert_models/
-儲存訓練模型至 chinese_models/
-儲存訓練模型至 roberta_models/

**✅ 模型評估與報告**
python evaluate.py

-使用測試集進行預測

-產出分類報告（accuracy、precision、recall、f1-score）

-輸出分類錯誤樣本供人工分析

**🌐 Streamlit Demo**
streamlit run app.py