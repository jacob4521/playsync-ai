from typing import Any
from typing import Dict
from typing import List
from fastapi import HTTPException
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai


# Load the data in .env to the memory
load_dotenv()

client = genai.Client()


class ChatRequest(BaseModel):
    prompt: str
    chatHistory: List[Dict[str, Any]] = []


class ProtectedChatRequest(BaseModel):
    prompt: str
    chatHistory: List[Dict[str, Any]] = []
    userId: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def search_playgrounds(lat: float, lon: float):
    """
    Search for available sports arenas or playgrounds based on GPS coordinates.
    When a user provides a city name (e.g., Galle, Colombo), convert it to
    latitude and longitude floats before calling this function.
    And also return the distance in the response.
    """

    # Call the node server for get arenas
    api_url = f"http://localhost:3000/arenas?lat={lat}&lon={lon}"
    response = requests.get(api_url)

    # Save that results response
    data = response.json()

    # Return data
    return data


import os
import requests


def get_user_bookings(user_id: str):
    """
    Fetch the list of bookings for a specific user using their user_id.
    This includes the arena name, date, time, and status.
    """
    # Node.js internal url
    api_url = f"http://localhost:3000/internal/bookings/{user_id}"

    # Get internal secret from .env
    internal_secret = os.getenv("INTERNAL_SERVER_KEY", "")
    headers = {"x-internal-secret": internal_secret}

    try:
        response = requests.get(api_url, headers=headers)
        return response.json()

    except Exception as e:
        print(f"Error fetching user bookings: {e}")
        return {"error": "Could not fetch bookings from the database"}


@app.post("/ai/assistant")
def chat_with_ai(request: ChatRequest):

    try:
        user_prompt = request.prompt

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=request.prompt,
            config={"tools": [search_playgrounds]},
        )

        # Send the response as json
        return {"message": response.text}

    except Exception as e:
        print(f"GenAI Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/ai/assistant/protected")
def chat_with_protected(request: ProtectedChatRequest):
    try:
        # Get the data
        user_prompt = request.prompt
        user_id = request.userId
        # user_chat_history = request.chatHistory

        # Prompt Injection: Get the user_id to AI
        enriched_prompt = (
            f"System Instruction: The current user's ID is '{user_id}'. "
            f"Always use this exact ID when calling tools like getting or adding bookings.\n\n"
            f"User Request: {user_prompt}"
        )

        # Send request to the Gemini Model
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=enriched_prompt,
            config={"tools": [get_user_bookings]},
        )

        # Send the response as json
        return {"message": response.text}

    except Exception as e:
        print(f"GenAI Protected Route Error: {e}")
        raise HTTPException(status_code=500, detail="Internal AI server error")
