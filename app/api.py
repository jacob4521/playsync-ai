import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Initialize the GenAI Client
client = genai.Client()

# Router
router = APIRouter()


# The Schema
class ChatRequest(BaseModel):
    prompt: str
    userId: str
    interaction_id: Optional[str] = None


# The POST Endpoints
@router.post("/chat")
def chat_with_ai(request: ChatRequest):
    try:
        if request.interaction_id:
            response = client.interactions.create(
                model="gemini-3.1-flash-lite",
                previous_interaction_id=request.interaction_id,
                input=request.prompt,
            )
        else:
            response = client.interactions.create(
                model="gemini-3.1-flash-lite", input=request.prompt
            )

        return {
            "status": "success",
            "interaction_id": response.id,
            "text": response.output_text,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
