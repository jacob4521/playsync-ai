import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from google import genai


# ---- System Instruction ----
SYSTEM_INSTRUCTION = """You are PlaySync AI, a helpful sports facility booking assistant.

CRITICAL RULES:
1. When users ask about available arenas, courts, or sports facilities, you MUST use the 'search_playgrounds' tool to fetch REAL data from our database.
2. You must ONLY present data returned by the tool. NEVER make up or hallucinate arena names, court details, or any facility information.
3. If the tool returns no results, tell the user that no facilities were found in that area.
4. Present the tool results in a clear, friendly, and well-formatted manner.
5. You can help users with booking-related questions and general sports information.
6. RULE: YOU MUST STRICTLY AND EXCLUSIVELY list ONLY the facilities returned by the tool. DO NOT add any stadiums, schools, or locations from your general knowledge. If the tool returns 2 results, your answer must ONLY contain those 2 results.
7. The answer should in the markdown table format and include all the data received from the function call.
8. CRITICAL RULE FOR BOOKINGS: When you use the 'check_bookings' tool, the array returned contains the times that are ALREADY BOOKED (Unavailable). You must explicitly tell the user that these specific times are UNAVAILABLE. 
9. TIME FORMATTING: The times from the database are in ISO format (e.g., 2026-06-18T02:13:41.058Z). Convert them into a readable format (e.g., "2:13 AM to 4:13 AM UTC") before showing them to the user.
"""

# Load environment variables
load_dotenv()

# Initialize the GenAI Client
client = genai.Client()


search_playgrounds_tool = {
    "type": "function",
    "name": "search_playgrounds",
    "description": "Search the database for available sports playgrounds based on location, sport type, and date.",
    "parameters": {
        "type": "object",
        "properties": {
            "lat": {
                "type": "number",
                "description": "The latitude of the target location (e.g., 6.0367 for Galle)",
            },
            "lon": {
                "type": "number",
                "description": "The longitude of the target location (e.g., 80.2170 for Galle)",
            },
        },
        "required": ["lat", "lon"],
    },
}


check_bookings_tool = {
    "type": "function",
    "name": "check_bookings",
    "description": "Check bookings of a court based on the given court_id and date.",
    "parameters": {
        "type": "object",
        "properties": {
            "court_id": {
                "type": "string",
                "description": "The id of the court",
            },
            "date": {
                "type": "string",
                "description": "The date, e.g. 2022-12-31",
            },
        },
        "required": ["court_id", "date"],
    },
}


# Router
router = APIRouter()


# The Schema
class ChatRequest(BaseModel):
    prompt: str
    userId: str
    interaction_id: Optional[str] = None


def sanitize_for_gemini(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_gemini(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        if len(obj) == 0:
            return "none"
        return [sanitize_for_gemini(item) for item in obj]

    return obj


# The POST Endpoints
@router.post("/chat")
def chat_with_ai(request: ChatRequest):
    try:
        if request.interaction_id:
            response = client.interactions.create(
                model="gemini-3.1-flash-lite",
                previous_interaction_id=request.interaction_id,
                input=request.prompt,
                tools=[search_playgrounds_tool, check_bookings_tool],
                system_instruction=SYSTEM_INSTRUCTION,
            )
        else:
            response = client.interactions.create(
                model="gemini-3.1-flash-lite",
                input=request.prompt,
                tools=[search_playgrounds_tool, check_bookings_tool],
                system_instruction=SYSTEM_INSTRUCTION,
            )

        # Check if the response contains a function call
        fc_step = next((s for s in response.steps if s.type == "function_call"), None)

        if fc_step and fc_step.name == "search_playgrounds":
            lat = fc_step.arguments.get("lat")
            lon = fc_step.arguments.get("lon")
            print(f"--> AI requested backend data for Lat: {lat}, Lon: {lon}")

            express_url = f"http://localhost:3000/arenas?lat={lat}&lon={lon}"

            try:
                express_response = httpx.get(express_url, timeout=10.0)
                express_response.raise_for_status()

                backend_data = express_response.json()
                print("--> Data received from Express:", backend_data)

            except Exception as e:
                print(f"--> Error connecting to Express: {e}")
                backend_data = {
                    "error": "Could not connect to the Express database or no data found."
                }

            sanitized_data = sanitize_for_gemini(backend_data)

            final_response = client.interactions.create(
                model="gemini-3.1-flash-lite",
                previous_interaction_id=response.id,
                input=[
                    {
                        "type": "function_result",
                        "call_id": fc_step.id,
                        "name": fc_step.name,
                        "result": sanitized_data,
                    }
                ],
                tools=[search_playgrounds_tool, check_bookings_tool],
                system_instruction=SYSTEM_INSTRUCTION,
            )

            return {
                "status": "success",
                "interaction_id": final_response.id,
                "text": final_response.output_text,
            }
        elif fc_step and fc_step.name == "check_bookings":
            court_id = fc_step.arguments.get("court_id")
            date = fc_step.arguments.get("date")
            print(f"--> AI checking availability for Court ID: {court_id} on {date}")

            express_url = f"http://localhost:3000/bookings/availability?courtId={court_id}&date={date}"

            try:
                express_response = httpx.get(express_url, timeout=10.0)
                express_response.raise_for_status()

                raw_data = express_response.json()
                backend_data = {"bookings": raw_data}

                print("--> Availability Data from Express:", backend_data)

            except Exception as e:
                print(f"--> Error connecting to Express: {e}")
                backend_data = {
                    "error": "Could not connect to the Express database or no data found."
                }

            sanitized_data = sanitize_for_gemini(backend_data)

            final_response = client.interactions.create(
                model="gemini-3.1-flash-lite",
                previous_interaction_id=response.id,
                input=[
                    {
                        "type": "function_result",
                        "call_id": fc_step.id,
                        "name": fc_step.name,
                        "result": sanitized_data,
                    }
                ],
                tools=[search_playgrounds_tool, check_bookings_tool],
                system_instruction=SYSTEM_INSTRUCTION,
            )

            return {
                "status": "success",
                "interaction_id": final_response.id,
                "text": final_response.output_text,
            }
        else:
            return {
                "status": "success",
                "interaction_id": response.id,
                "text": response.output_text,
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
