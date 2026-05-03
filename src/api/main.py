from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.tasks.celery_app import process_text_task
from celery.result import AsyncResult

app = FastAPI(title="Async ML Rubert Service")

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: PredictRequest):
    """Принимает текст, кладет задачу в очередь и сразу возвращает ID"""
    task = process_text_task.delay(request.text)
    return {
        "task_id": task.id, 
        "message": "Текст принят в обработку. Используйте /result/{task_id} для получения ответа."
    }

@app.get("/result/{task_id}")
def get_result(task_id: str):
    """Проверяет статус задачи и возвращает результат, если готово"""
    result = AsyncResult(task_id)
    
    if result.ready():
        if result.failed():
            raise HTTPException(status_code=500, detail="Ошибка при обработке модели")
        return {"status": "completed", "data": result.result}
    
    return {"status": "processing", "message": "Модель еще считает..."}