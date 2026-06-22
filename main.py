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


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def search_playgrounds(lat: float, lon: float):
    """
    Search for available sports arenas or playgrounds based on GPS coordinates.
    When a user provides a city name (e.g., Galle, Colombo), convert it to
    latitude and longitude floats before calling this function.
    """

    # Call the node server for get arenas
    api_url = f"http://localhost:3000/arenas?lat={lat}&lon={lon}"
    response = requests.get(api_url)

    # Save that results response
    data = response.json()

    # Return data
    return data


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
