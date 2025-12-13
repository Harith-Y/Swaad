"""
Pydantic models and data structures for the Swaad API.
"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class DishInput(BaseModel):
    name: str
    category: str


class UserDishes(BaseModel):
    dishes: List[DishInput]


class MenuText(BaseModel):
    text: str
    restaurant_name: Optional[str] = None


class FlavorProfile(BaseModel):
    appetizer: Dict[str, float]
    mains: Dict[str, float]
    desserts: Dict[str, float]


class UserProfile(BaseModel):
    allergies: Optional[List[str]] = None
    favorite_dishes: Optional[List[DishInput]] = None
    diet_type: Optional[str] = None


class PreferencePrompt(BaseModel):
    prompt: str


class DummyUserUpdate(BaseModel):
    location: Optional[str] = None
    allergies: Optional[List[str]] = None
    favorite_dishes: Optional[List[Dict[str, str]]] = None
    diet_type: Optional[str] = None


class RecommendationsRequest(BaseModel):
    user_dishes: List[DishInput]
    limit: int = 10


class RecipeRecommendation(BaseModel):
    name: str
    ingredients: List[str]
    instructions: str
    similarity_score: float
    flavor_profile: Dict[str, float]


class RecommendationsResponse(BaseModel):
    recommendations: List[RecipeRecommendation]


class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    location: Optional[str] = None
    max_results: Optional[int] = None
    diet_type: Optional[str] = None
    user_key: Optional[str] = "default"


class RestaurantRecommendation(BaseModel):
    id: str
    name: str
    url: Optional[str]
    avg_rating: Optional[float]
    price_range: Optional[int]
    cuisine_types: Optional[List[str]]
    location: Optional[Dict[str, Any]]
    coordinates: Optional[Dict[str, float]]
    menu_items: List[str]
    popular_dishes: Optional[List[str]]
    taste_vector: List[float]
    recommended_dishes: List[Dict[str, Any]]
    photos: Optional[List[str]]
    menu_url: Optional[str]
    score: float


class ChatResponse(BaseModel):
    response: Dict[str, str]
    chat_id: Optional[str]
    menu_buddy: Dict[str, Any]

