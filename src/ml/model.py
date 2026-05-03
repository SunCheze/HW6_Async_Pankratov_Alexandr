import os
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = os.getenv("MODEL_NAME", "cointegrated/rubert-tiny2")

# ==========================================
# PATTER: WARM LOADING
# Код в этом блоке выполнится ОДИН РАЗ при старти воркера.
# ==========================================
print(f"[ML Model] Начинаю загрузку {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print("[ML Model] Модель успешно загружена и готова к работе!")

def get_embedding(text: str) -> list:
    """Синхронная функция получения эмбеддинга"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Возвращаем весь вектор (в реальном проде можно вернуть только первые 5 чисел для экономии трафика)
    return outputs.last_hidden_state[:, 0, :].numpy().tolist()[0]