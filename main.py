from typing import Optional
from typing import Any
from typing import Dict
from typing import List
from fastapi import HTTPException
import requests
import json
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
    userId: str
    interactionId: Optional[str] = None


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

get_user_bookings_tool = {
    "type": "function",
    "name": "get_user_bookings",
    "description": "Fetch the list of bookings for a specific user.",
    "parameters": {
        "type": "object",
        "properties": {
            "userId": {"type": "string", "description": "The ID of the user"}
        },
        "required": ["userId"],
    },
}


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
        # Get the data from the req body
        user_prompt = request.prompt
        user_id = request.userId
        interaction_id = request.interactionId  # optional

        # Prompt Injection: Get the user_id to AI
        enriched_prompt = (
            f"System Instruction: The current user's ID is '{user_id}'. "
            f"Always use this exact ID when calling tools like getting or adding bookings.\n\n"
            f"User Request: {user_prompt}"
        )

        # Config for the gemini request
        ai_request_params = {
            "model": "gemini-3.1-flash-lite",
            "input": enriched_prompt,
            "tools": [get_user_bookings_tool],
        }

        if interaction_id:
            ai_request_params["previous_interaction_id"] = interaction_id

        # Send the request to gemini
        interaction = client.interactions.create(**ai_request_params)

        # Check if the interaction contains steps
        if interaction.steps:
            # Iterate for steps
            for step in interaction.steps:
                # If the step type is function call and the name is get_user_bookings
                if step.type == "function_call" and step.name == "get_user_bookings":
                    # Get the user_id to search
                    uid_to_search = step.arguments.get("userId")

                    # Execute the function
                    print(f"Executing tool for user: {uid_to_search}")
                    booking_data = get_user_bookings(uid_to_search)

                    # Return the booking data to AI
                    interaction = client.interactions.create(
                        model="gemini-3.1-flash-lite",
                        tools=[get_user_bookings_tool],
                        previous_interaction_id=interaction.id,
                        input=[
                            {
                                "type": "function_result",
                                "call_id": step.id,
                                "name": step.name,
                                "result": [
                                    {"type": "text", "text": json.dumps(booking_data)}
                                ],
                            }
                        ],
                    )
                    break

        # Return the final response to the user
        return {"message": interaction.output_text, "interactionId": interaction.id}

    except Exception as e:
        print(f"GenAI Protected Route Error: {e}")
        raise HTTPException(status_code=500, detail="Internal AI server error")
