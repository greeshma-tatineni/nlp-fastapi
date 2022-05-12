from typing import Optional

from fastapi import FastAPI
from transformers import pipeline
from fastapi.responses import FileResponse
#from requestmodel import Request
import uvicorn
from pydantic import BaseModel

class Request(BaseModel):
    context: str 
    question: str 

app = FastAPI()

@app.post('/predict')
def predict(data:Request):
    data = data.dict()
    question = data['question']
    context = data['context']

    print(question)
    print(context)

    question_answer = pipeline("question-answering", model = "huggingface-course/bert-finetuned-squad")
    return question_answer(question=question, context=context)

@app.get("/")
def read_root():
    return "Welcome to Q and A"

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)