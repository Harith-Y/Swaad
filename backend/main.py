"""
Swaad Recipe Recommendation API - Main Application
Modularized version with clean separation of concerns.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS, SENTENCE_TRANSFORMER_MODEL, GROQ_API_KEY
from embeddings import get_embedding_model
from dish_processing import get_groq_client
from models import ChatRequest
from routes.chat import chat_endpoint


# Initialize FastAPI app
app = FastAPI(title="Swaad Recipe Recommendation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_models():
    """Preload models on startup to avoid first-request delay."""
    print(f"Preloading sentence-transformer model: {SENTENCE_TRANSFORMER_MODEL}")
    get_embedding_model()
    print("Sentence-transformer model loaded.")
    
    if GROQ_API_KEY:
        get_groq_client()
        print("Groq client initialized.")


@app.get("/")
def read_root():
    """Root endpoint - API health check."""
    return {"message": "Swaad Recipe Recommendation API is running"}


@app.post("/api/chat")
async def chat_with_restaurants(request: ChatRequest):
    """
    Chat endpoint for restaurant recommendations.
    
    This endpoint:
    1. Takes a natural language query from the user
    2. Uses Pinecone for semantic search of restaurants
    3. Ranks results based on taste similarity and user preferences
    4. Returns personalized restaurant and dish recommendations
    """
    return await chat_endpoint(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

