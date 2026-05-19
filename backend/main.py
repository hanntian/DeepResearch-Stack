from fastapi import FastAPI
from pydantic import BaseModel

from retrieval.qa_pipeline import generate_answer

app = FastAPI()


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/ask")
def ask(request: AskRequest):
    result = generate_answer(
        query=request.query,
        top_k=request.top_k,
    )
    return result
import requests

