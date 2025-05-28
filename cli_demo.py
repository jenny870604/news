import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "./bert_models/roberta-news"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
label_map = {0: "中央社",1: "中港澳",2: "公共政策",3: "公民運動",4: "品味生活",
            5: "國內",6: "國際",7: "政治",8: "科技",9: "評論",10: "調查",11: "財經",12: "風生活"}

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return label_map[pred], probs[0][pred].item()

if __name__ == "__main__":
    text = input("請輸入新聞：")
    label, confidence = classify(text)
    print(f"➡️ 分類結果：{label}（信心度：{confidence:.2f}）")
