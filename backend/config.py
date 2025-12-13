"""
Configuration and environment variables for the Swaad API.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "menu-buddy")

# Model Configuration
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# Feature Flags
USE_SEMANTIC_DISH_TASTE = os.getenv("USE_SEMANTIC_DISH_TASTE", "true").lower() in {"1", "true", "yes", "y"}
USE_SEMANTIC_INGREDIENT_TASTE = os.getenv("USE_SEMANTIC_INGREDIENT_TASTE", "true").lower() in {"1", "true", "yes", "y"}

# Default User Settings
DEFAULT_USER_LOCATION = os.getenv("DEFAULT_USER_LOCATION", "")

# File Paths
INGREDIENT_FLAVOR_CSV = "ingredient-flavor.csv"

# Taste Vector Configuration
TASTE_DIMENSIONS = ["sweet", "salty", "sour", "bitter", "umami", "spicy"]
TASTE_VECTOR_SIZE = 6

# Recommendation Settings
DEFAULT_MAX_RESULTS = 10
TASTE_SIMILARITY_WEIGHT = 0.35
FAVORITES_BOOST_WEIGHT = 0.1

# CORS Settings
CORS_ORIGINS = ["*"]  # Configure as needed for production

