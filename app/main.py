from sys import prefix
from fastapi import FastAPI
from app.api import router as api_router


# Initialize the FastAPI app
app = FastAPI(
    title="playsync AI Service",
    description="The AI Booking Assistant Microservice for PlaySync",
)


# Basic health check route
@app.get("/")
def read_root():
    return {"status": "success", "message": "PLaysync AI Service is running!"}


# Include APIs
app.include_router(api_router, prefix="/api")