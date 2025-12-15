"""
User data management and dummy user storage.
"""
from typing import Dict, Any, List, Optional
from models import UserProfile, DishInput
from config import DEFAULT_USER_LOCATION


# In-memory user storage (replace with real database in production)
USER_METADATA_MAP = {
    "dummy2": {
        "location": "Chicago",
        "allergies": ["Nuts", "Cabbage", "Cauliflower"],
        "favorite_dishes": [
            {"name": "Paneer butter masala", "category": "mains"},
            {"name": "Veg Manchurian", "category": "appetizer"},
            {"name": "Peppy Paneer Pizza", "category": "mains"},
        ],
        "diet_type": "veg",
        "flavor_profile": {
            "appetizer": {"spicy": 0.6, "sweet": 0.2, "umami": 0.7, "sour": 0.3, "salty": 0.6},
            "mains": {"spicy": 0.7, "sweet": 0.2, "umami": 0.8, "sour": 0.3, "salty": 0.7},
            "desserts": {"spicy": 0.1, "sweet": 0.8, "umami": 0.2, "sour": 0.2, "salty": 0.2},
            "allergies": ["Nuts", "Cabbage", "Cauliflower"],
            "favorite_dishes": [
                {"name": "Paneer butter masala", "category": "mains"},
                {"name": "Veg Manchurian", "category": "appetizer"},
                {"name": "Peppy Paneer Pizza", "category": "mains"},
            ],
            "diet_type": "veg",
        },
        "pending_query": None,
    },
    "default": {
        "location": "New York, NY",
        "allergies": ["shellfish"],
        "favorite_dishes": [
            {"name": "Pepperoni Pizza", "category": "mains"},
            {"name": "Chicken Wings", "category": "appetizer"},
            {"name": "Cheesecake", "category": "desserts"}
        ],
        "diet_type": "non-veg",
        "flavor_profile": {
            "appetizer": {"spicy": 0.8, "sweet": 0.1, "umami": 0.7, "sour": 0.2, "salty": 0.8},
            "mains": {"spicy": 0.6, "sweet": 0.2, "umami": 0.8, "sour": 0.2, "salty": 0.8},
            "desserts": {"spicy": 0.1, "sweet": 0.9, "umami": 0.2, "sour": 0.2, "salty": 0.2},
            "allergies": ["shellfish"],
            "favorite_dishes": [
                {"name": "Pepperoni Pizza", "category": "mains"},
                {"name": "Chicken Wings", "category": "appetizer"},
                {"name": "Cheesecake", "category": "desserts"}
            ],
            "diet_type": "non-veg",
        },
        "pending_query": None,
    },
    "dummy3": {
        "location": "San Francisco, CA",
        "allergies": ["gluten"],
        "favorite_dishes": [
            {"name": "Spicy Tuna Roll", "category": "mains"},
            {"name": "Tonkotsu Ramen", "category": "mains"},
            {"name": "Mango Mochi", "category": "desserts"}
        ],
        "diet_type": "mix",
        "flavor_profile": {
            "appetizer": {"spicy": 0.5, "sweet": 0.2, "umami": 0.7, "sour": 0.4, "salty": 0.6},
            "mains": {"spicy": 0.7, "sweet": 0.2, "umami": 0.9, "sour": 0.3, "salty": 0.7},
            "desserts": {"spicy": 0.1, "sweet": 0.8, "umami": 0.3, "sour": 0.3, "salty": 0.2},
            "allergies": ["gluten"],
            "favorite_dishes": [
                {"name": "Spicy Tuna Roll", "category": "mains"},
                {"name": "Tonkotsu Ramen", "category": "mains"},
                {"name": "Mango Mochi", "category": "desserts"}
            ],
            "diet_type": "mix",
        },
        "pending_query": None,
    }
}


def normalize_favorite_dishes(favorite_dishes: Any) -> List[Dict[str, str]]:
    """Normalize favorite dishes to a consistent format."""
    if not favorite_dishes:
        return []
    result = []
    for d in favorite_dishes:
        if isinstance(d, dict):
            result.append({"name": d.get("name", ""), "category": d.get("category", "")})
        elif hasattr(d, "name") and hasattr(d, "category"):
            result.append({"name": d.name, "category": d.category})
    return result


def get_dummy_user(user_key: str = "default") -> Dict[str, Any]:
    """Get dummy user data by key."""
    if user_key not in USER_METADATA_MAP:
        USER_METADATA_MAP[user_key] = {
            "location": "",
            "allergies": [],
            "favorite_dishes": [],
            "diet_type": "mix",
            "flavor_profile": {},
            "pending_query": None,
        }
    return USER_METADATA_MAP[user_key]


def dummy_user_to_user_profile(u: Dict[str, Any]) -> Optional[UserProfile]:
    """Convert dummy user dict to UserProfile model."""
    if not u:
        return None
    allergies = u.get("allergies")
    favorite_dishes_raw = u.get("favorite_dishes")
    diet_type = u.get("diet_type")
    
    favorite_dishes = None
    if favorite_dishes_raw:
        favorite_dishes = [
            DishInput(name=d.get("name", ""), category=d.get("category", ""))
            for d in favorite_dishes_raw
            if isinstance(d, dict)
        ]
    
    return UserProfile(
        allergies=allergies,
        favorite_dishes=favorite_dishes,
        diet_type=diet_type
    )


def sync_dummy_user_from_request(request: Any) -> None:
    """Sync user metadata from request to dummy user storage."""
    user_key = (request.user_key or "default").strip() or "default"
    dummy_user = get_dummy_user(user_key)
    
    if request.location:
        dummy_user["location"] = request.location
    
    if hasattr(request, "allergies") and request.allergies is not None:
        dummy_user["allergies"] = request.allergies
    
    if hasattr(request, "favorite_dishes") and request.favorite_dishes is not None:
        dummy_user["favorite_dishes"] = normalize_favorite_dishes(request.favorite_dishes)
    
    if request.diet_type:
        dummy_user["diet_type"] = request.diet_type


def update_dummy_user(user_key: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Update dummy user data."""
    dummy_user = get_dummy_user(user_key)
    
    if data.get("location") is not None:
        dummy_user["location"] = data["location"]
    if data.get("allergies") is not None:
        dummy_user["allergies"] = data["allergies"]
    if data.get("favorite_dishes") is not None:
        dummy_user["favorite_dishes"] = normalize_favorite_dishes(data["favorite_dishes"])
    if data.get("diet_type") is not None:
        dummy_user["diet_type"] = data["diet_type"]
    
    return dummy_user

