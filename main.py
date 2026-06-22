from fastapi import FastAPI
from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ai/assistant")
def chat_with_ai(request: ChatRequest):

    user_prompt = request.prompt

    return {"received_prompt": user_prompt}
