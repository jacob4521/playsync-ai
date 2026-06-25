from typing import Optional
from typing import Any
from typing import Dict
from typing import List
from fastapi import HTTPException
import requests
import json
import os
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
    userId: str
    interactionId: Optional[str] = None


app = FastAPI()


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

check_availability_tool = {
    "type": "function",
    "name": "check_availability",
    "description": (
        "Fetch the list of booked slots for a specific court on a specific date. "
        "CRITICAL INSTRUCTION: You MUST ALWAYS call search_playgrounds FIRST to get the "
        "list of arenas and courts, even if you think you already know the court ID from "
        "a previous conversation. Court IDs can change, so NEVER reuse a court ID from "
        "memory or previous context. After calling search_playgrounds, look inside the "
        "'courts' array of the relevant arena in the JSON response, find the exact "
        "matching court name, and extract its exact 'id'. Pass that freshly extracted "
        "exact 'id' to this tool as the courtId. NEVER guess or reuse old IDs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "courtId": {"type": "string", "description": "The ID of the court. MUST be obtained from a fresh search_playgrounds call in this conversation turn."},
            "date": {"type": "string", "description": "The date in YYYY-MM-DD format"},
        },
        "required": ["courtId", "date"],
    },
}

search_playgrounds_tool = {
    "type": "function",
    "name": "search_playgrounds",
    "description": (
        "Search for available sports arenas or playgrounds based on GPS lat and lon coordinates. "
        "ALWAYS call this tool first before calling check_availability, even in a continuing conversation, "
        "because you need the exact court 'id' from the response to pass as 'courtId'. "
        "If the user provides a city name (e.g., Galle, Colombo), convert it to latitude and longitude floats. "
        "Look inside the 'courts' array of the returned arena to find the court by name and extract its 'id'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "The latitude of the location"},
            "lon": {"type": "number", "description": "The longitude of the location"},
        },
        "required": ["lat", "lon"],
    },
}


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


def check_availability(courtId: str, date: str):
    """
    Fetch the list of booked slots for a specific court on specific date.
    """
    # Node.js internal url
    api_url = (
        f"http://localhost:3000/internal/availability?courtId={courtId}&date={date}"
    )

    # Get internal secret from .env
    internal_secret = os.getenv("INTERNAL_SERVER_KEY", "")
    headers = {"x-internal-secret": internal_secret}

    try:
        response = requests.get(api_url, headers=headers)
        print(f"[check_availability] courtId={courtId} date={date} -> HTTP {response.status_code}")
        print(f"[check_availability] response body: {response.text[:300]}")

        if response.status_code == 422:
            return {"error": "Invalid courtId or date format. The courtId must be a valid court ID obtained from search_playgrounds."}

        return response.json()

    except Exception as e:
        print(f"Error checking availability: {e}")
        return {"error": "Could not fetch availability from the database"}


@app.get("/")
def read_root():
    return {"Hello": "World"}


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

        # Always inject system context (userId + instructions) on EVERY message.
        # Even when continuing a conversation (interactionId exists), the AI needs
        # fresh instructions to avoid using stale court IDs from previous context.
        system_context = json.dumps({"currentUserId": user_id})
        enriched_prompt = (
            f"[SYSTEM CONTEXT]: {system_context}\n"
            f"Instruction: The 'currentUserId' above is the logged-in user's ID. "
            f"You MUST pass it as the 'userId' argument ONLY when calling the "
            f"'get_user_bookings' tool. Do NOT pass userId to any other tool "
            f"(e.g. check_availability or search_playgrounds do NOT take a userId). "
            f"Do NOT invent or modify the userId value.\n"
            f"IMPORTANT: When checking court availability, you MUST ALWAYS call "
            f"search_playgrounds FIRST to get fresh court IDs. NEVER reuse court IDs "
            f"from previous messages in this conversation.\n\n"
            f"User Request: {user_prompt}"
        )


        all_tools = [
            get_user_bookings_tool,
            check_availability_tool,
            search_playgrounds_tool,
        ]

        # Config for the gemini request
        ai_request_params = {
            "model": "gemini-3.1-flash-lite",
            "input": enriched_prompt,
            "tools": all_tools,
        }

        if interaction_id:
            ai_request_params["previous_interaction_id"] = interaction_id

        # Send the request to gemini
        interaction = client.interactions.create(**ai_request_params)

        processed_step_ids = set()
        # Cache actual tool results so duplicate calls with same args
        # return the REAL data instead of a misleading "No data found".
        tool_result_cache: Dict[str, Any] = {}
        MAX_TOOL_CALLS = 10  # hard safety cap across all tool calls

        while len(processed_step_ids) < MAX_TOOL_CALLS:
            has_tool_call = False

            if interaction.steps:
                for step in interaction.steps:
                    if (
                        step.type == "function_call"
                        and step.id not in processed_step_ids
                    ):
                        processed_step_ids.add(step.id)
                        has_tool_call = True

                        # Build a cache key: tool name + its primary arguments
                        if step.name == "get_user_bookings":
                            cache_key = f"get_user_bookings:{step.arguments.get('userId')}"
                        elif step.name == "check_availability":
                            cache_key = f"check_availability:{step.arguments.get('courtId')}:{step.arguments.get('date')}"
                        elif step.name == "search_playgrounds":
                            cache_key = f"search_playgrounds:{step.arguments.get('lat')}:{step.arguments.get('lon')}"
                        else:
                            cache_key = step.name

                        # If we already ran this exact tool+args, return cached result
                        # to stop retry loops without misleading the AI with fake data.
                        if cache_key in tool_result_cache:
                            tool_data = tool_result_cache[cache_key]
                            print(
                                f"[CACHE HIT] Returning cached result for '{step.name}' "
                                f"({cache_key}) to stop retry loop."
                            )
                        else:
                            # First time: actually execute the tool and cache the result
                            tool_data = None

                            if step.name == "get_user_bookings":
                                uid_to_search = step.arguments.get("userId")
                                print(f"Executing get_user_bookings for user: {uid_to_search}")
                                tool_data = get_user_bookings(uid_to_search)

                            elif step.name == "search_playgrounds":
                                lat = step.arguments.get("lat")
                                lon = step.arguments.get("lon")
                                print(f"Executing search_playgrounds for lat: {lat} and lon: {lon}")
                                tool_data = search_playgrounds(lat, lon)

                            elif step.name == "check_availability":
                                court_id_to_search = step.arguments.get("courtId")
                                date_to_search = step.arguments.get("date")
                                print(
                                    f"Executing check_availability for court: {court_id_to_search} on {date_to_search}"
                                )
                                tool_data = check_availability(court_id_to_search, date_to_search)

                            tool_result_cache[cache_key] = tool_data

                        # Send the tool result back to the AI
                        interaction = client.interactions.create(
                            model="gemini-3.1-flash-lite",
                            tools=all_tools,
                            previous_interaction_id=interaction.id,
                            input=[
                                {
                                    "type": "function_result",
                                    "call_id": step.id,
                                    "name": step.name,
                                    "result": [
                                        {"type": "text", "text": json.dumps(tool_data)}
                                    ],
                                }
                            ],
                        )
                        break

            if not has_tool_call:
                break


        return {"message": interaction.output_text, "interactionId": interaction.id}

    except Exception as e:
        print(f"GenAI Protected Route Error: {e}")
        raise HTTPException(status_code=500, detail="Internal AI server error")
