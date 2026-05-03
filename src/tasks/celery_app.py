import os
from celery import Celery
from src.ml.model import get_embedding

# Инициализация Celery. Берем URL из переменных окружения Docker-контейнера
app = Celery(
    'ml_worker',
    broker=os.getenv('BROKER_URL'),
    backend=os.getenv('RESULT_BACKEND')
)

@app.task(bind=True, max_retries=1)
def process_text_task(self, text: str):
    """
    Задача, которая выполняется в фоне.
    Забирает текст, прогоняет через модель, возвращает результат обратно в брокер.
    """
    try:
        # Вызываем Warm-загруженную модель
        embedding = get_embedding(text)
        
        # Для демонстрации возвращаем превью (первые 5 элементов вектора), 
        # чтобы не засорять консоль тысячами чисел
        return {
            "status": "success",
            "text_length": len(text),
            "embedding_preview": embedding[:5] 
        }
    except Exception as e:
        # Базовая обработка ошибок в Celery
        self.retry(exc=e)