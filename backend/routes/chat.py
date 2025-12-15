"""
Chat endpoint for restaurant recommendations.
"""
from fastapi import HTTPException
from typing import Dict, Any, Optional, Tuple
import json
import re
import csv
import os

from models import ChatRequest
from database import get_dummy_user, sync_dummy_user_from_request, dummy_user_to_user_profile
from embeddings import embed_text, combine_vectors, get_embedding_model
from taste_analysis import user_profile_to_taste_vector, infer_taste_from_text_hybrid
from pinecone_client import get_pinecone_index, maybe_upsert_ingredients_to_pinecone
from recommendations import filter_and_rank_recommendations
from dish_processing import get_groq_client, classify_dish_diet_with_groq
from config import GROQ_API_KEY, USE_SEMANTIC_INGREDIENT_TASTE
from recipe_database import (
    load_recipes_database,
    search_recipe_by_name,
    get_taste_vector_from_recipe,
    has_valid_taste_profile
)


# Load all ingredients from CSV at startup (cached globally)
_INGREDIENT_LIST = None

def load_ingredients_from_csv() -> list[str]:
    """Load all ingredient names from ingredient-flavor.csv"""
    global _INGREDIENT_LIST
    
    if _INGREDIENT_LIST is not None:
        return _INGREDIENT_LIST
    
    ingredients = []
    csv_path = os.path.join(os.path.dirname(__file__), "..", "ingredient-flavor.csv")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ingredient = row.get('ingredient', '').strip().lower()
                if ingredient:
                    ingredients.append(ingredient)
        
        _INGREDIENT_LIST = ingredients
        print(f"[INFO] Loaded {len(ingredients)} ingredients from CSV")
        return ingredients
    except Exception as e:
        print(f"[ERROR] Failed to load ingredients from CSV: {e}")
        # Fallback to basic list
        _INGREDIENT_LIST = ['chicken', 'paneer', 'ginger', 'garlic', 'tomato', 'onion']
        return _INGREDIENT_LIST


def extract_location_from_query(query: str) -> Optional[str]:
    """
    Extract location from the query text.

    Returns:
        Location string if found, None otherwise
    """
    query_lower = query.lower()

    # Location patterns
    patterns = [
        r'\bnear\s+(.+?)(?:\s*,|\s+also|\s+and|\s*$)',
        r'\bin\s+(.+?)(?:\s*,|\s+also|\s+and|\s*$)',
        r'\bat\s+(.+?)(?:\s*,|\s+also|\s+and|\s*$)',
        r'\baround\s+(.+?)(?:\s*,|\s+also|\s+and|\s*$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            location = match.group(1).strip()
            # Filter out common non-location words
            non_locations = ["me", "here", "there", "my place", "home"]
            if location not in non_locations and len(location) > 2:
                return location

    return None


def extract_cuisine_from_query(query: str) -> Optional[str]:
    """
    Extract cuisine type from the query text.

    Returns:
        Cuisine string if found (e.g., 'Thai', 'Italian', 'Chinese'), None otherwise
    """
    query_lower = query.lower()

    # Common cuisine types with their variations
    cuisines = {
        'thai': ['thai'],
        'italian': ['italian', 'pizza', 'pasta'],
        'chinese': ['chinese'],
        'japanese': ['japanese', 'sushi', 'ramen'],
        'indian': ['indian', 'curry'],
        'mexican': ['mexican', 'taco', 'burrito'],
        'korean': ['korean', 'kimchi', 'bibimbap'],
        'vietnamese': ['vietnamese', 'pho'],
        'american': ['american', 'burger'],
        'french': ['french'],
        'greek': ['greek'],
        'mediterranean': ['mediterranean'],
        'middle eastern': ['middle eastern', 'falafel', 'shawarma'],
        'spanish': ['spanish', 'tapas', 'paella'],
    }

    # Pattern: look for cuisine keywords in query
    # Examples: "Thai food", "Italian restaurant", "I want Chinese"
    for cuisine_name, keywords in cuisines.items():
        for keyword in keywords:
            # Check if keyword appears as a whole word
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query_lower):
                print(f"[DEBUG] Cuisine type detected in query: {cuisine_name}")
                return cuisine_name

    return None


def extract_diet_from_query(query: str) -> Optional[str]:
    """
    Extract diet preference from the query text.

    This function checks:
    1. Explicit diet mentions (e.g., "I want veg food")
    2. Dish-based diet detection (e.g., "I want chicken curry" -> non-veg)

    Returns:
        'veg', 'vegetarian', 'non-veg', or None if not specified
    """
    query_lower = query.lower()

    # STEP 1: Check for explicit diet preference
    # Vegetarian indicators
    veg_patterns = [
        r'\b(?:i am|i\'m|looking for|want|need|prefer)\s+(?:a\s+)?veg(?:etarian)?\s+(?:food|dish|meal|option)',
        r'\bveg(?:etarian)?\s+(?:\w+\s+)?(?:food|dish|meal|option|restaurant|cuisine)',  # Allows "vegetarian Thai food"
        r'\bonly\s+veg(?:etarian)?',
        r'\bpure\s+veg(?:etarian)?',
        r'\bvegetarian\s+only',
        r'\bno\s+(?:meat|non-veg|nonveg)',
        r'\bplant[-\s]?based',
    ]

    # Non-vegetarian indicators
    nonveg_patterns = [
        r'\b(?:i am|i\'m|looking for|want|need|prefer)\s+(?:a\s+)?non[-\s]?veg(?:etarian)?\s+(?:food|dish|meal|option)',
        r'\bnon[-\s]?veg(?:etarian)?\s+(?:\w+\s+)?(?:food|dish|meal|option|restaurant|cuisine)',  # Allows "non-veg Thai food"
        r'\b(?:meat|chicken|fish|seafood)\s+(?:lover|eater)',
        r'\bonly\s+non[-\s]?veg',
    ]

    # Check for explicit vegetarian
    for pattern in veg_patterns:
        if re.search(pattern, query_lower):
            print(f"[DEBUG] Explicit veg preference detected in query")
            return "veg"

    # Check for explicit non-vegetarian
    for pattern in nonveg_patterns:
        if re.search(pattern, query_lower):
            print(f"[DEBUG] Explicit non-veg preference detected in query")
            return "non-veg"

    # STEP 2: Check for dish-based diet detection
    # Extract potential dish names from query
    dish_patterns = [
        # Specific pattern for "to eat/have/try X" - must come first
        r'(?:i want|i need)\s+to\s+(?:eat|have|try)\s+(?:some\s+)?(.+?)(?:\s+(?:near|in|at|and)|$)',
        # "I want to X" (where X is dish, not "to eat") - captures the dish after "to"
        r'(?:i want|i need)\s+to\s+(?!(?:eat|have|try)\s)(?:some\s+)?(.+?)(?:\s+(?:near|in|at|and)|$)',
        # General patterns (exclude "to" prefix)
        r'(?:i want|i need|looking for|get me|find|show me|give me)\s+(?!to\s)(?:some\s+)?(.+?)(?:\s+(?:near|in|at|and)|$)',
        r'(?:where.*get|where.*find)\s+(.+?)(?:\s+(?:near|in|at|and)|$)',
        r'(?:is there|do you have|any)\s+(.+?)(?:\s+(?:available|near|in|at)|$)',
    ]

    detected_dishes = []
    for pattern in dish_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            # Clean up the match
            dish = match.strip()
            # Remove common words
            dish = re.sub(r'\b(a|an|the|some|any|place|restaurant|where|that|has)\b', '', dish).strip()
            
            # Filter out generic/preference-only terms (same as is_dish_query)
            if dish and len(dish) > 2:
                # Check if it's just preferences/generic terms
                generic_terms = ["food", "something", "anything", "restaurant", "place", "restaurants", "places"]
                preference_words = ["spicy", "savory", "sweet", "sour", "salty", "bitter", "hot", "mild", 
                                   "delicious", "tasty", "good", "fresh", "healthy", "light", "heavy"]
                
                words = dish.split()
                meaningful_words = [w for w in words if w not in (generic_terms + preference_words + ["and", "or"])]
                
                # Only add if there are meaningful dish words
                if len(meaningful_words) > 0 and dish not in generic_terms:
                    detected_dishes.append(dish)

    # Classify detected dishes
    if detected_dishes:
        print(f"[DEBUG] Detected potential dishes in query: {detected_dishes}")

        veg_count = 0
        nonveg_count = 0

        for dish in detected_dishes:
            classification = classify_dish_diet_with_groq(dish)
            if classification == "veg":
                veg_count += 1
            else:
                nonveg_count += 1

        # If all dishes are veg, return veg
        if veg_count > 0 and nonveg_count == 0:
            print(f"[DEBUG] All detected dishes are vegetarian -> diet: veg")
            return "veg"

        # If all dishes are non-veg, return non-veg
        if nonveg_count > 0 and veg_count == 0:
            print(f"[DEBUG] All detected dishes are non-vegetarian -> diet: non-veg")
            return "non-veg"

        # If mixed, return None (use user profile default)
        if veg_count > 0 and nonveg_count > 0:
            print(f"[DEBUG] Mixed veg/non-veg dishes detected -> using user profile default")
            return None

    return None


def search_dish_in_db(dish_name: str) -> Optional[Dict[str, Any]]:
    """
    Search for a dish using hybrid approach:
    1. Search Pinecone ingredients namespace
    2. Search recipe CSV database (231K recipes)
    3. Return None if not found

    Returns:
        Dict with dish info if found, None otherwise
    """
    # STEP 1: Search in Pinecone ingredients namespace
    try:
        pc_index = get_pinecone_index()
        model = get_embedding_model()

        # Create embedding for dish name
        dish_embedding = model.encode(dish_name).tolist()

        # Search in ingredients namespace first
        result = pc_index.query(
            vector=dish_embedding,
            top_k=5,
            include_metadata=True,
            namespace="ingredients"
        )

        matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])

        # Check if we have a good match (score > 0.8)
        if matches and len(matches) > 0:
            best_match = matches[0]
            score = best_match.get("score", 0) if isinstance(best_match, dict) else getattr(best_match, "score", 0)

            if score > 0.8:
                metadata = best_match.get("metadata") if isinstance(best_match, dict) else getattr(best_match, "metadata", {})
                print(f"[DEBUG] Found dish '{dish_name}' in Pinecone with score {score}")
                return {
                    "found": True,
                    "source": "pinecone",
                    "dish_name": metadata.get("name", dish_name),
                    "taste_vector": [
                        metadata.get("sweet", 0),
                        metadata.get("salty", 0),
                        metadata.get("sour", 0),
                        metadata.get("bitter", 0),
                        metadata.get("umami", 0),
                        metadata.get("spicy", 0)
                    ],
                    "ingredients": metadata.get("ingredients", []),
                    "metadata": metadata
                }

    except Exception as e:
        print(f"[ERROR] Failed to search dish in Pinecone: {e}")

    # STEP 2: Search in recipe CSV database (231K recipes)
    try:
        recipe = search_recipe_by_name(dish_name, threshold=0.6)

        if recipe and has_valid_taste_profile(recipe):
            taste_vector = get_taste_vector_from_recipe(recipe)
            print(f"[DEBUG] Found dish '{dish_name}' in recipe database: '{recipe['original_name']}'")
            return {
                "found": True,
                "source": "csv",
                "dish_name": recipe["original_name"],
                "taste_vector": taste_vector,
                "ingredients": recipe.get("ingredients", []),
                "metadata": {
                    "name": recipe["original_name"],
                    "ingredients": recipe.get("ingredients", []),
                    "recipe_id": recipe.get("id", "")
                }
            }
        elif recipe and not has_valid_taste_profile(recipe):
            print(f"[DEBUG] Found dish '{dish_name}' in CSV but has zero taste vector, will use Groq")
            return None

    except Exception as e:
        print(f"[ERROR] Failed to search dish in recipe database: {e}")

    print(f"[DEBUG] Dish '{dish_name}' not found in any database")
    return None


def get_ingredients_from_groq(dish_name: str) -> Optional[Dict[str, Any]]:
    """
    Get ingredients and taste profile for a dish using Groq API.

    Returns:
        Dict with ingredients and taste vector, or None if failed
    """
    try:
        groq_client = get_groq_client()

        prompt = f"""Analyze the dish "{dish_name}" and provide:
1. Main ingredients (comma-separated list)
2. Taste profile on a scale of 0-1 for each: sweet, salty, sour, bitter, umami, spicy

Respond in this exact JSON format:
{{
    "dish_name": "{dish_name}",
    "ingredients": ["ingredient1", "ingredient2", ...],
    "taste_profile": {{
        "sweet": 0.0-1.0,
        "salty": 0.0-1.0,
        "sour": 0.0-1.0,
        "bitter": 0.0-1.0,
        "umami": 0.0-1.0,
        "spicy": 0.0-1.0
    }}
}}

Only respond with valid JSON, nothing else."""

        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500
        )

        response_text = completion.choices[0].message.content.strip()

        # Try to parse JSON
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        dish_info = json.loads(response_text)

        print(f"[DEBUG] Got ingredients from Groq for '{dish_name}': {dish_info.get('ingredients', [])}")

        return dish_info

    except Exception as e:
        print(f"[ERROR] Failed to get ingredients from Groq: {e}")
        return None


def save_dish_to_db(dish_name: str, dish_info: Dict[str, Any]) -> bool:
    """
    Save dish and its ingredients to Pinecone database.

    Args:
        dish_name: Name of the dish
        dish_info: Dict containing ingredients and taste_profile from Groq

    Returns:
        True if successful, False otherwise
    """
    try:
        pc_index = get_pinecone_index()
        model = get_embedding_model()

        taste_profile = dish_info.get("taste_profile", {})
        ingredients = dish_info.get("ingredients", [])

        # Create embedding for dish
        dish_embedding = model.encode(dish_name).tolist()

        # Create taste vector
        taste_vector = [
            float(taste_profile.get("sweet", 0)),
            float(taste_profile.get("salty", 0)),
            float(taste_profile.get("sour", 0)),
            float(taste_profile.get("bitter", 0)),
            float(taste_profile.get("umami", 0)),
            float(taste_profile.get("spicy", 0))
        ]

        # Prepare vectors to upsert
        vectors = []

        # Add dish vector
        dish_vector = {
            "id": f"ingredient:{dish_name.lower().replace(' ', '_')}",
            "values": dish_embedding,
            "metadata": {
                "type": "ingredient",
                "name": dish_name,
                "sweet": taste_vector[0],
                "salty": taste_vector[1],
                "sour": taste_vector[2],
                "bitter": taste_vector[3],
                "umami": taste_vector[4],
                "spicy": taste_vector[5],
                "ingredients": ingredients[:10]  # Limit to 10 ingredients
            }
        }
        vectors.append(dish_vector)

        # Add ingredient vectors (if not already in DB)
        for ingredient in ingredients[:10]:  # Limit to 10 ingredients
            ingredient_embedding = model.encode(ingredient).tolist()

            # Use average taste profile for individual ingredients
            # (In a real system, you'd want to get specific taste profiles for each ingredient)
            ingredient_vector = {
                "id": f"ingredient:{ingredient.lower().replace(' ', '_')}",
                "values": ingredient_embedding,
                "metadata": {
                    "type": "ingredient",
                    "name": ingredient,
                    "sweet": taste_vector[0] * 0.5,  # Reduced weight for individual ingredients
                    "salty": taste_vector[1] * 0.5,
                    "sour": taste_vector[2] * 0.5,
                    "bitter": taste_vector[3] * 0.5,
                    "umami": taste_vector[4] * 0.5,
                    "spicy": taste_vector[5] * 0.5,
                }
            }
            vectors.append(ingredient_vector)

        # Upsert to Pinecone
        pc_index.upsert(vectors=vectors, namespace="ingredients")

        print(f"[DEBUG] Saved dish '{dish_name}' and {len(ingredients)} ingredients to DB")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to save dish to DB: {e}")
        return False


def is_greeting(query: str) -> bool:
    """
    Check if the query is a greeting.

    Returns:
        True if query is a greeting, False otherwise
    """
    query_lower = query.lower().strip()
    greetings = [
        "hi", "hello", "hey", "hola", "greetings", "good morning",
        "good afternoon", "good evening", "howdy", "what's up",
        "whats up", "sup", "yo", "hiya", "heya"
    ]

    # Check if query is exactly a greeting or starts with greeting
    for greeting in greetings:
        if query_lower == greeting or query_lower.startswith(greeting + " "):
            return True

    return False


def normalize_dish_name_with_groq(dish_name: str) -> str:
    """
    Use Groq to normalize/correct misspelled or partial dish names.
    
    Args:
        dish_name: Raw dish name extracted from query (may be misspelled)
        
    Returns:
        Normalized dish name or original if Groq fails
    """
    try:
        groq_client = get_groq_client()
        
        prompt = f"""You are a food name correction assistant. Given a potentially misspelled or partial dish name, return the correct, most likely full dish name.

Examples:
- "pizz" -> "pizza"
- "burgr" -> "burger"  
- "pasta alfrd" -> "pasta alfredo"
- "panr tikka" -> "paneer tikka"
- "biryani" -> "biryani" (already correct)

Dish name: "{dish_name}"

Respond with ONLY the corrected dish name, nothing else. If the name seems correct, return it as is."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        
        corrected_name = response.choices[0].message.content.strip().lower()
        
        # Clean up the response (remove quotes, extra spaces)
        corrected_name = corrected_name.strip('"\'').strip()
        
        if corrected_name and corrected_name != dish_name.lower():
            print(f"[DEBUG] Groq corrected '{dish_name}' -> '{corrected_name}'")
            return corrected_name
        
        return dish_name
        
    except Exception as e:
        print(f"[ERROR] Groq normalization failed: {e}")
        return dish_name


def is_dish_query(query: str) -> Optional[str]:
    """
    Check if the query is asking for a specific dish (not at a specific restaurant).

    Handles both single and multiple dishes (e.g., "paratha and paneer curry").

    Returns:
        Dish name(s) if detected (comma-separated if multiple), None otherwise
    """
    query_lower = query.lower().strip()

    # Patterns for dish queries - more specific patterns
    patterns = [
        # "is there a place where X is available"
        r"is\s+there\s+(?:a\s+)?(?:place|restaurant)\s+(?:where|that\s+has)\s+(.+?)\s+(?:is\s+)?available",
        # "where can I find X" or "where can I get X" or "where i can get X"
        r"where\s+(?:can\s+)?i\s+(?:can\s+)?(?:find|get)\s+(.+?)(?:\s+near|\s+in|\s+at|\s*$)",
        # "do you have X" or "is there X"
        r"^(?:do\s+you\s+have|is\s+there)\s+(?:any\s+)?(.+?)(?:\s+available|\s+near|\s+in|\s+at|\s*$)",
        # "I want to eat/have/try X" - specific pattern to skip "to eat/have/try"
        r"^(?:i\s+want|i'd\s+like|i\s+need)\s+to\s+(?:eat|have|try)\s+(?:some\s+)?(.+?)(?:\s+near|\s+in|\s+at|\s*$)",
        # "I want to X" (where X is dish, not "to eat") - captures the dish after "to"
        r"^(?:i\s+want|i'd\s+like|i\s+need)\s+to\s+(?!(?:eat|have|try)\s)(?:some\s+)?(.+?)(?:\s+near|\s+in|\s+at|\s*$)",
        # "I want X" or "show me X" (no "to" involved)
        r"^(?:i\s+want|i'd\s+like|i\s+need|give\s+me|show\s+me|find|get\s+me|looking\s+for)\s+(?!to\s)(?:some\s+)?(.+?)(?:\s+near|\s+in|\s+at|\s*$)",
        # "can I get X"
        r"^can\s+i\s+get\s+(.+?)(?:\s+near|\s+in|\s+at|\s*$)",
    ]

    # Don't treat greetings or very short queries as dish queries
    if len(query_lower.split()) <= 1:
        return None

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            dish_name = match.group(1).strip()
            # Clean up the dish name
            dish_name = re.sub(r'\s+(?:near|in|at|from)\s+.*$', '', dish_name).strip()
            # Remove articles and common words
            dish_name = re.sub(r'^(?:a|an|the|some)\s+', '', dish_name).strip()
            
            # Filter out generic/vague terms that aren't dish names
            generic_terms = ["food", "something", "anything", "restaurant", "place", "restaurants", "places"]
            
            # Filter out preference descriptors (not actual dish names)
            preference_words = ["spicy", "savory", "sweet", "sour", "salty", "bitter", "hot", "mild", 
                               "delicious", "tasty", "good", "fresh", "healthy", "light", "heavy"]
            
            # Check if dish_name contains only generic terms and preferences
            words = dish_name.split()
            meaningful_words = [w for w in words if w not in (generic_terms + preference_words + ["and", "or"])]
            
            # If no meaningful dish name words remain, it's not a dish query
            if len(meaningful_words) == 0:
                return None  # It's just preferences/generic terms, not a dish name
            
            # Also reject if the whole phrase is a generic term
            if dish_name in generic_terms:
                return None
            
            if len(dish_name) > 2:
                return dish_name

    return None


def extract_ingredients_from_query(query: str) -> list[str]:
    """
    Extract ingredient names mentioned in the query.
    Uses the comprehensive ingredient list from ingredient-flavor.csv (622 ingredients).
    
    Returns:
        List of ingredient names found in query
    """
    query_lower = query.lower()
    
    # Load all ingredients from CSV (cached after first call)
    all_ingredients = load_ingredients_from_csv()
    
    found_ingredients = []
    
    # Check for each ingredient in the query
    # Sort by length (descending) to match longer phrases first
    # e.g., "soy sauce" before "soy"
    sorted_ingredients = sorted(all_ingredients, key=len, reverse=True)
    
    for ingredient in sorted_ingredients:
        # Use word boundaries to avoid partial matches
        # e.g., "ham" shouldn't match "graham"
        pattern = r'\b' + re.escape(ingredient) + r'\b'
        if re.search(pattern, query_lower):
            found_ingredients.append(ingredient)
    
    return found_ingredients


def is_restaurant_menu_query(query: str) -> Optional[str]:
    """
    Check if the query is asking for a specific restaurant's menu.

    Returns:
        Restaurant name if detected, None otherwise
    """
    query_lower = query.lower().strip()

    # Patterns for restaurant menu queries
    patterns = [
        r"(?:what'?s|show|tell me|get)\s+(?:the\s+)?menu\s+(?:for|of|at)\s+(.+?)(?:\s+restaurant)?$",
        r"menu\s+(?:for|of|at)\s+(.+?)(?:\s+restaurant)?$",
        r"(?:show|tell)\s+me\s+(.+?)(?:'s|\s+)menu",
        r"what\s+does\s+(.+?)\s+(?:have|serve|offer)",
        r"what\s+can\s+i\s+(?:get|order)\s+(?:at|from)\s+(.+?)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            restaurant_name = match.group(1).strip()
            # Clean up common words
            restaurant_name = re.sub(r'\s+restaurant$', '', restaurant_name).strip()
            if len(restaurant_name) > 2:
                return restaurant_name

    return None


def parse_specific_query(query: str) -> Optional[Tuple[str, str]]:
    """
    Parse query to detect if user is asking about a specific dish at a specific restaurant.

    Returns:
        Tuple of (dish_name, restaurant_name) if specific query detected, None otherwise
    """
    query_lower = query.lower()

    # Pattern: "dish ... where restaurant is X" or "dish ... at/from X"
    pattern1 = r"(.+?)\s+where\s+restaurant\s+is\s+(.+?)$"
    match = re.search(pattern1, query_lower)
    if match:
        dish = match.group(1).strip()
        restaurant = match.group(2).strip()
        # Clean up common prefixes
        dish = re.sub(r"^(?:i want|i'd like|give me|show me|find)\s+", "", dish).strip()
        dish = re.sub(r"\s+like\s+", " ", dish).strip()  # Remove "like"
        return (dish, restaurant)

    return None


async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """
    Main chat endpoint for restaurant recommendations.
    Uses Pinecone for semantic search and taste-based ranking.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    groq_client = get_groq_client()
    
    # Ensure ingredients are loaded into Pinecone
    maybe_upsert_ingredients_to_pinecone()

    # Sync user metadata from request
    sync_dummy_user_from_request(request)
    user_key = (request.user_key or "default").strip() or "default"
    dummy_user = get_dummy_user(user_key)
    dummy_profile = dummy_user_to_user_profile(dummy_user)

    # Extract user preferences
    allergies = (dummy_user.get("allergies") or []) if isinstance(dummy_user, dict) else []
    favorite_dishes = (dummy_user.get("favorite_dishes") or []) if isinstance(dummy_user, dict) else []
    diet_type = (dummy_user.get("diet_type") if isinstance(dummy_user, dict) else None) or "mix"

    print(f"[DEBUG] Using dummy user: {user_key}")
    print(f"[DEBUG] Allergies: {allergies}")
    print(f"[DEBUG] Favorite dishes: {favorite_dishes}")
    print(f"[DEBUG] Diet type: {diet_type}")

    if dummy_profile:
        allergies = dummy_profile.allergies or allergies
        favorite_dishes = [d.model_dump() for d in (dummy_profile.favorite_dishes or [])] or favorite_dishes
        diet_type = dummy_profile.diet_type or diet_type

    # Extract diet preference from query (overrides user profile)
    query_diet = extract_diet_from_query(request.query)
    if query_diet:
        diet_type = query_diet
        print(f"[DEBUG] Diet type overridden from query: {diet_type}")

    # Extract cuisine type from query
    query_cuisine = extract_cuisine_from_query(request.query)

    # Extract location from query (PRIORITY: query location overrides everything)
    query_location = extract_location_from_query(request.query)
    if query_location:
        # Query location takes highest priority
        request.location = query_location
        print(f"[DEBUG] Location extracted from query (PRIORITY): {query_location}")
        
        # If location is generic (downtown, nearby, etc.) and user has home city, contextualize it
        generic_locations = ["downtown", "nearby", "near me", "around here", "local", "in the area"]
        if any(generic in query_location.lower() for generic in generic_locations):
            user_db_location = (dummy_user.get("location") if isinstance(dummy_user, dict) else None) or ""
            if user_db_location:
                # Append user's city to make it specific
                request.location = f"{query_location}, {user_db_location}"
                print(f"[DEBUG] Contextualized generic location '{query_location}' with user city: {request.location}")
    elif not query_location and not request.location:
        # No location in query or request, will use fallback later
        print(f"[DEBUG] No location in query, will use fallback location")

    # Calculate user taste vector
    user_taste_vec = user_profile_to_taste_vector(dummy_profile) if dummy_profile else [0.0] * 6
    print(f"[DEBUG] Initial user_taste_vec from profile: {[round(x, 2) for x in user_taste_vec]}")
    
    fav_text = ""
    if favorite_dishes:
        try:
            fav_text = " ".join([
                (d.name if hasattr(d, "name") else (d.get("name") if isinstance(d, dict) else str(d)))
                for d in favorite_dishes
            ])
        except Exception:
            fav_text = ""
    
    print(f"[DEBUG] Favorite dishes text for taste inference: '{fav_text}'")
    inferred_user = infer_taste_from_text_hybrid(fav_text, semantic=USE_SEMANTIC_INGREDIENT_TASTE)
    print(f"[DEBUG] Inferred taste vector: {[round(x, 2) for x in inferred_user]}")
    user_taste_vec = combine_vectors(user_taste_vec, inferred_user, secondary_weight=0.35)
    print(f"[DEBUG] Final combined user_taste_vec: {[round(x, 2) for x in user_taste_vec]}")

    # Handle location and pending queries
    is_first_turn = not request.chat_id
    pending_query = dummy_user.get("pending_query") if isinstance(dummy_user, dict) else None
    if is_first_turn and pending_query and request.query and not request.location:
        request.location = request.query
        request.query = pending_query
        dummy_user["location"] = request.location
        dummy_user["pending_query"] = None
    
    user_db_location = (dummy_user.get("location") if isinstance(dummy_user, dict) else None) or ""
    fallback_location = request.location
    if not fallback_location and user_db_location:
        fallback_location = user_db_location
    
    print(f"[DEBUG] user_key: {user_key}")
    print(f"[DEBUG] user_db_location: {user_db_location}")
    print(f"[DEBUG] request.location: {request.location}")
    print(f"[DEBUG] fallback_location: {fallback_location}")

    # Set max results
    final_max_results = request.max_results or 10
    print(f"[DEBUG] max_results set to: {final_max_results}")

    # Initialize flags
    dish_not_found = False
    dish_not_found_name = None

    # Check if this is a greeting
    if is_greeting(request.query):
        print(f"[DEBUG] Greeting detected")
        greeting_responses = [
            "Hello! 👋 I'm Swaad, your AI food companion. I can help you discover amazing restaurants and dishes based on your taste preferences. What are you craving today?",
            "Hey there! 🍽️ Welcome to Swaad! Tell me what kind of food you're in the mood for, and I'll find the perfect restaurants for you.",
            "Hi! 😊 I'm here to help you find delicious food. Whether you're craving something spicy, sweet, or savory, just let me know and I'll recommend the best spots!",
        ]
        import random
        return {
            "response": {
                "text": random.choice(greeting_responses)
            },
            "chat_id": request.chat_id,
            "menu_buddy": {"recommendations": []}
        }

    # Check if query is relevant to food/restaurants
    from dish_processing import is_relevant_query
    if not is_relevant_query(request.query):
        print(f"[DEBUG] Non-food query detected: '{request.query}'")
        out_of_scope_responses = [
            "I'm Swaad, your food and restaurant recommendation assistant! 🍽️ I specialize in helping you discover amazing dishes and restaurants. For questions about tourist attractions, hotels, or other travel info, you might want to check a travel guide. But if you're hungry and looking for great food recommendations, I'm here to help! What are you craving?",
            "I focus on food and restaurant recommendations! 😊 While I can't help with general travel or tourist spots, I'm excellent at finding delicious restaurants and dishes that match your taste. Want to explore some amazing food options instead?",
            "That's outside my expertise! I'm your AI food companion, specialized in restaurant and dish recommendations. 🍴 If you're looking for places to eat or specific dishes to try, I'm your go-to assistant! What kind of food are you in the mood for?"
        ]
        import random
        return {
            "response": {
                "text": random.choice(out_of_scope_responses)
            },
            "chat_id": request.chat_id,
            "menu_buddy": {"recommendations": []}
        }

    # Check if this is a restaurant menu query
    restaurant_name_query = is_restaurant_menu_query(request.query)
    if restaurant_name_query:
        print(f"[DEBUG] Restaurant menu query detected: '{restaurant_name_query}'")

        # Search for the restaurant in Pinecone
        pc_index = get_pinecone_index()
        model = get_embedding_model()

        # Create embedding for restaurant name
        restaurant_embedding = model.encode(restaurant_name_query).tolist()

        # Search in restaurants namespace
        result = pc_index.query(
            vector=restaurant_embedding,
            top_k=5,
            include_metadata=True,
            namespace="restaurants"
        )

        matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])

        # Find the best matching restaurant
        best_match = None
        best_score = 0

        for match in matches:
            meta = match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", {})
            score = match.get("score", 0) if isinstance(match, dict) else getattr(match, "score", 0)
            rest_name = meta.get("name", "")

            # Check if restaurant name matches
            if restaurant_name_query.lower() in rest_name.lower() or rest_name.lower() in restaurant_name_query.lower():
                if score > best_score:
                    best_score = score
                    best_match = meta

        if best_match:
            # Found the restaurant - return its full menu
            menu_items = best_match.get("menu_items", [])

            # Format menu items as recommended dishes
            recommended_dishes = []
            for item in menu_items[:20]:  # Limit to 20 items
                recommended_dishes.append({
                    "name": item,
                    "similarity": 50.0  # Neutral similarity since we're showing the full menu (50%)
                })

            return {
                "response": {
                    "text": f"Here's the menu for {best_match.get('name')}:"
                },
                "chat_id": request.chat_id,
                "menu_buddy": {
                    "recommendations": [{
                        "name": best_match.get("name"),
                        "rating": best_match.get("avg_rating"),
                        "price_range": best_match.get("price_range"),
                        "cuisine_types": best_match.get("cuisine_types", []),
                        "recommended_dishes": recommended_dishes
                    }]
                }
            }
        else:
            # Restaurant not found
            return {
                "response": {
                    "text": f"Sorry, I couldn't find a restaurant named '{restaurant_name_query}' in our database. Could you try a different name or ask for dish recommendations instead?"
                },
                "chat_id": request.chat_id,
                "menu_buddy": {"recommendations": []}
            }

    # Check if this is a dish-specific query (not at a specific restaurant)
    dish_query = is_dish_query(request.query)
    if dish_query and "where restaurant is" not in request.query.lower():
        # Normalize dish name with Groq (correct typos, partial names)
        normalized_dish = normalize_dish_name_with_groq(dish_query)
        print(f"[DEBUG] Dish-specific query detected: '{dish_query}'" + 
              (f" -> normalized to '{normalized_dish}'" if normalized_dish != dish_query else ""))
        
        # Use normalized name for search
        dish_query = normalized_dish

        # STEP 1: Search for dish in our database
        dish_in_db = search_dish_in_db(dish_query)

        # STEP 2: If not found, get ingredients from Groq and save to DB
        if not dish_in_db:
            print(f"[DEBUG] Dish '{dish_query}' not in DB, calling Groq API...")
            dish_info_from_groq = get_ingredients_from_groq(dish_query)

            if dish_info_from_groq:
                # Save to database
                save_dish_to_db(dish_query, dish_info_from_groq)

                # Create dish_in_db structure from Groq response
                taste_profile = dish_info_from_groq.get("taste_profile", {})
                dish_in_db = {
                    "found": True,
                    "dish_name": dish_query,
                    "taste_vector": [
                        float(taste_profile.get("sweet", 0)),
                        float(taste_profile.get("salty", 0)),
                        float(taste_profile.get("sour", 0)),
                        float(taste_profile.get("bitter", 0)),
                        float(taste_profile.get("umami", 0)),
                        float(taste_profile.get("spicy", 0))
                    ],
                    "ingredients": dish_info_from_groq.get("ingredients", []),
                    "metadata": {
                        "name": dish_query,
                        "ingredients": dish_info_from_groq.get("ingredients", [])
                    }
                }

        # STEP 3: Search for restaurants that have this dish
        try:
            pc_index = get_pinecone_index()
            # Search all restaurants for this dish
            all_restaurants = pc_index.query(
                vector=[0.0] * 384,  # Dummy vector to get all restaurants
                top_k=100,
                include_metadata=True,
                namespace="restaurants"
            )
            matches = all_restaurants.get("matches", []) if isinstance(all_restaurants, dict) else getattr(all_restaurants, "matches", [])

            # Search for restaurants that have this dish
            restaurants_with_dish = []
            for m in matches:
                meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
                menu_items = meta.get("menu_items", [])

                # Check if dish exists in menu
                for menu_item in menu_items:
                    if dish_query.lower() in menu_item.lower() or menu_item.lower() in dish_query.lower():
                        restaurants_with_dish.append({
                            "name": meta.get("name"),
                            "rating": meta.get("avg_rating"),
                            "price_range": meta.get("price_range"),
                            "cuisine_types": meta.get("cuisine_types", []),
                            "dish": menu_item,
                            "metadata": meta
                        })
                        break  # Only add restaurant once

            if not restaurants_with_dish:
                # Dish not found in restaurants - but we have dish info from DB/Groq
                print(f"[DEBUG] Dish '{dish_query}' not found in any restaurant")

                if dish_in_db:
                    # We have dish info, use it for taste-based recommendations
                    print(f"[DEBUG] Using dish taste profile for recommendations")
                    dish_taste_vec = dish_in_db.get("taste_vector", [0.0] * 6)
                    # Combine user taste with dish taste
                    user_taste_vec = combine_vectors(user_taste_vec, dish_taste_vec, secondary_weight=0.6)

                # Set flag to modify response text
                dish_not_found = True
                dish_not_found_name = dish_query
            else:
                # Dish found - return restaurants that have it
                print(f"[DEBUG] Found '{dish_query}' at {len(restaurants_with_dish)} restaurants")

                # Apply location filtering
                from dish_processing import check_location_match
                if fallback_location:
                    print(f"[DEBUG] Applying location filter to dish results: {fallback_location}")
                    filtered_restaurants = []
                    for rest in restaurants_with_dish:
                        loc_json = rest["metadata"].get("location_json", "{}")
                        try:
                            import json
                            location = json.loads(loc_json) if isinstance(loc_json, str) else loc_json
                            loc_str = location
                            if isinstance(location, dict):
                                parts = []
                                for key in ["address", "city", "state", "zip_code", "country"]:
                                    if key in location and location[key]:
                                        parts.append(str(location[key]))
                                if parts:
                                    loc_str = ", ".join(parts)
                                else:
                                    loc_str = ", ".join([str(v) for v in location.values() if isinstance(v, (str, int))])
                            
                            if check_location_match(fallback_location, str(loc_str)):
                                print(f"[DEBUG] Location match: {rest['name']} in {loc_str}")
                                filtered_restaurants.append(rest)
                            else:
                                print(f"[DEBUG] Filtered out {rest['name']} - wrong location: {loc_str}")
                        except Exception as e:
                            print(f"[DEBUG] Error checking location for {rest['name']}: {e}")
                            continue
                    
                    restaurants_with_dish = filtered_restaurants
                    print(f"[DEBUG] After location filter: {len(restaurants_with_dish)} restaurants")

                # Rank by rating and taste similarity
                from recommendations import dish_recommendations_for_restaurant
                ranked_restaurants = []
                for rest in restaurants_with_dish[:10]:  # Limit to top 10
                    # Get the matched dish
                    matched_dish = rest.get("dish", dish_query)

                    # Get all recommended dishes (including the matched one)
                    menu_items = rest["metadata"].get("menu_items", [])
                    all_dishes = dish_recommendations_for_restaurant(
                        menu_items=menu_items,
                        user_taste_vec=user_taste_vec,
                        diet_type=diet_type,
                        allergies=allergies,
                        top_n=10  # Get more to ensure we have the matched dish
                    )

                    # Find the matched dish in the recommendations (it will have proper similarity)
                    recommended_dishes = []
                    matched_dish_obj = None
                    
                    for dish in all_dishes:
                        dish_name = dish.get("name") if isinstance(dish, dict) else dish
                        if dish_name.lower() == matched_dish.lower():
                            matched_dish_obj = dish
                            break
                    
                    # Put matched dish first (with its calculated similarity)
                    if matched_dish_obj:
                        recommended_dishes.append(matched_dish_obj)
                    else:
                        # Fallback if not found in recommendations
                        recommended_dishes.append({"name": matched_dish, "similarity": 50.0})
                    
                    # Add other dishes
                    for dish in all_dishes:
                        dish_name = dish.get("name") if isinstance(dish, dict) else dish
                        if dish_name.lower() != matched_dish.lower():
                            recommended_dishes.append(dish)
                            if len(recommended_dishes) >= 5:  # Limit to 5 total
                                break

                    ranked_restaurants.append({
                        "name": rest["name"],
                        "rating": rest["rating"],
                        "price_range": rest["price_range"],
                        "cuisine_types": rest["cuisine_types"],
                        "recommended_dishes": recommended_dishes
                    })

                # Sort by rating
                ranked_restaurants.sort(key=lambda x: x.get("rating", 0), reverse=True)

                return {
                    "response": {
                        "text": f"Great choice! I found '{dish_query}' at {len(restaurants_with_dish)} restaurants. Here are the top-rated ones:"
                    },
                    "chat_id": request.chat_id,
                    "menu_buddy": {"recommendations": ranked_restaurants[:final_max_results]}
                }

        except Exception as e:
            print(f"[DEBUG] Error in dish query: {e}")
            import traceback
            traceback.print_exc()

    # Check if this is a specific query (dish at specific restaurant)
    specific_query = parse_specific_query(request.query)
    if specific_query:
        dish_name, restaurant_name = specific_query
        print(f"[DEBUG] Specific query detected: dish='{dish_name}', restaurant='{restaurant_name}'")

        # Search for the specific restaurant and dish
        try:
            pc_index = get_pinecone_index()
            # Search for the restaurant
            restaurant_vec = embed_text(restaurant_name)
            query_res = pc_index.query(vector=restaurant_vec, top_k=20, include_metadata=True, namespace="restaurants")
            matches = query_res.get("matches", []) if isinstance(query_res, dict) else getattr(query_res, "matches", [])

            # Find the matching restaurant
            target_restaurant = None
            for m in matches:
                meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
                rest_name = meta.get("name", "").lower()
                if restaurant_name.lower() in rest_name or rest_name in restaurant_name.lower():
                    target_restaurant = meta
                    target_restaurant["id"] = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
                    break

            if not target_restaurant:
                return {
                    "response": {
                        "text": f"Sorry, I couldn't find a restaurant named '{restaurant_name}' in our database."
                    },
                    "chat_id": request.chat_id,
                    "menu_buddy": {"recommendations": []}
                }

            # Find the specific dish in the restaurant's menu
            menu_items = target_restaurant.get("menu_items", [])
            matching_dish = None
            for dish in menu_items:
                if dish_name.lower() in dish.lower() or dish.lower() in dish_name.lower():
                    matching_dish = dish
                    break

            if not matching_dish:
                return {
                    "response": {
                        "text": f"Sorry, '{dish_name}' is not available at {target_restaurant.get('name')}. Available dishes: {', '.join(menu_items[:5])}"
                    },
                    "chat_id": request.chat_id,
                    "menu_buddy": {"recommendations": []}
                }

            # Return minimal response with just the dish and restaurant
            avg_rating = target_restaurant.get("avg_rating", "N/A")
            return {
                "response": {
                    "text": f"{matching_dish} is available at {target_restaurant.get('name')} (Rating: {avg_rating}/5)"
                },
                "chat_id": request.chat_id,
                "menu_buddy": {
                    "recommendations": [{
                        "restaurant_name": target_restaurant.get("name"),
                        "dish_name": matching_dish,
                        "rating": avg_rating,
                        "price_range": target_restaurant.get("price_range"),
                        "location": target_restaurant.get("location")
                    }]
                }
            }
        except Exception as e:
            print(f"[DEBUG] Error in specific query: {e}")
            import traceback
            traceback.print_exc()

    # Check if location is provided (for general queries)
    if is_first_turn and not fallback_location:
        if isinstance(dummy_user, dict):
            dummy_user["pending_query"] = request.query
        return {
            "response": {
                "text": "Please share your location (city or ZIP/postal code) so I can find restaurants near you."
            },
            "chat_id": request.chat_id
        }

    # Query Pinecone for restaurant recommendations
    print(f"[DEBUG] Querying Pinecone for restaurant recommendations")

    # Initialize response structure
    if dish_not_found:
        # Check if multiple dishes were mentioned (contains "and" or ",")
        if " and " in dish_not_found_name or "," in dish_not_found_name:
            initial_text = f"Sorry, I couldn't find '{dish_not_found_name}' at our partner restaurants. However, based on your preferences and the taste profile of these dishes, here are some recommendations you might enjoy:"
        else:
            initial_text = f"Sorry, I couldn't find '{dish_not_found_name}' at our partner restaurants. However, based on your preferences and the taste profile of this dish, here are some recommendations you might enjoy:"
    else:
        initial_text = f"Here are some great restaurant recommendations for you in {fallback_location}!"

    ai_json = {
        "response": {
            "text": initial_text
        },
        "chat_id": request.chat_id
    }

    ranked = []
    try:
        pc_index = get_pinecone_index()
        qvec = embed_text(request.query)
        
        # IMPORTANT: When filtering by location, fetch MORE results since many will be filtered out
        # Use location from query if available, otherwise use fallback_location
        location_to_filter = query_location if query_location else fallback_location
        
        # Increase top_k significantly when location filter is active
        if location_to_filter:
            top_k = max(final_max_results * 10, 50)  # Fetch 10x more to account for location filtering
            print(f"[DEBUG] Location filter active, fetching top_k={top_k} (will filter to {final_max_results})")
        else:
            top_k = max(final_max_results, 10)
            
        print(f"[DEBUG] Querying Pinecone with top_k={top_k}")
        query_res = pc_index.query(vector=qvec, top_k=top_k, include_metadata=True, namespace="restaurants")
        matches = query_res.get("matches", []) if isinstance(query_res, dict) else getattr(query_res, "matches", [])
        print(f"[DEBUG] Pinecone returned {len(matches)} matches")
        
        # Extract ingredients from query for ingredient-based boosting
        query_ingredients = extract_ingredients_from_query(request.query)
        if query_ingredients:
            print(f"[DEBUG] Detected ingredients in query: {query_ingredients}")

        # Filter and rank recommendations
        if location_to_filter:
            print(f"[DEBUG] Applying location filter: {location_to_filter}")
        else:
            print(f"[DEBUG] No location filter applied")

        ranked = filter_and_rank_recommendations(
            matches=matches,
            user_taste_vec=user_taste_vec,
            favorite_dishes=favorite_dishes,
            diet_type=diet_type,
            allergies=allergies,
            max_results=final_max_results,
            query_text=request.query,
            location_filter=location_to_filter,
            cuisine_filter=query_cuisine,
            query_ingredients=query_ingredients
        )

        print(f"[DEBUG] Total ranked restaurants: {len(ranked)}")
        
        # If no results with cuisine filter, retry without it
        if len(ranked) == 0 and query_cuisine:
            print(f"[DEBUG] No {query_cuisine} restaurants found, retrying without cuisine filter")
            ranked = filter_and_rank_recommendations(
                matches=matches,
                user_taste_vec=user_taste_vec,
                favorite_dishes=favorite_dishes,
                diet_type=diet_type,
                allergies=allergies,
                max_results=final_max_results,
                query_text=request.query,
                location_filter=location_to_filter,
                cuisine_filter=None  # Retry without cuisine filter
            )
            print(f"[DEBUG] Found {len(ranked)} restaurants without cuisine filter")
            
            # Update the response message to inform user
            if len(ranked) > 0:
                location_name = location_to_filter or "your area"
                diet_label = "vegetarian " if diet_type == 'veg' else ""
                initial_text = f"I couldn't find any {query_cuisine.title()} restaurants in {location_name}, but here are some other great {diet_label}options nearby:"
                ai_json["response"]["text"] = initial_text
        
        print(f"[DEBUG] Returning top {len(ranked)} recommendations")
        
    except Exception as e:
        print(f"[DEBUG] Error in ranking: {e}")
        import traceback
        traceback.print_exc()
        ranked = []

    # Check if no results found
    if len(ranked) == 0:
        if dish_not_found:
            no_results_msg = f"Sorry, I couldn't find '{dish_not_found_name}' or any similar dishes at our partner restaurants"
            if location_to_filter:
                no_results_msg += f" in {location_to_filter}"
            if allergies:
                no_results_msg += f" that are safe for your allergies ({', '.join(allergies)})"
            no_results_msg += ". Try a different location or dish!"
            ai_json["response"]["text"] = no_results_msg
        else:
            no_results_msg = "Sorry, I couldn't find any restaurants matching your preferences"
            if location_to_filter:
                no_results_msg += f" in {location_to_filter}"
            if diet_type and diet_type != "mix":
                no_results_msg += f" with {diet_type} options"
            if allergies:
                no_results_msg += f" that are safe for your allergies"
            no_results_msg += ". Try broadening your search or changing your location!"
            ai_json["response"]["text"] = no_results_msg
    
    # Filter response to only include essential fields
    filtered_recommendations = []
    for restaurant in ranked:
        filtered_restaurant = {
            "name": restaurant.get("name"),
            "rating": restaurant.get("avg_rating"),
            "price_range": restaurant.get("price_range"),
            "cuisine_types": restaurant.get("cuisine_types", []),
            "recommended_dishes": [
                {
                    "name": dish.get("name"),
                    "similarity": dish.get("similarity")
                }
                for dish in restaurant.get("recommended_dishes", [])[:5]  # Limit to top 5 dishes
            ]
        }
        filtered_recommendations.append(filtered_restaurant)

    ai_json["menu_buddy"] = {
        "recommendations": filtered_recommendations
    }
    
    # Rewrite response text to match filtered results
    try:
        original_text = ai_json.get("response", {}).get("text", "")
        if original_text and ranked:
            restaurant_names = [r.get("name") for r in ranked if r.get("name")]
            diet_label = "vegetarian" if diet_type in {"veg", "vegetarian"} else diet_type or "any diet"

            if dish_not_found:
                rewrite_prompt = f"""The user asked for '{dish_not_found_name}' but we don't have it at our partner restaurants.
Rewrite this text to:
1. Apologize that '{dish_not_found_name}' is not available at our partner restaurants
2. Mention these {len(ranked)} alternative restaurants: {', '.join(restaurant_names[:5])}
3. Say these are similar recommendations based on their taste preferences
4. Only mention dishes suitable for {diet_label} diet
5. Keep the tone friendly and helpful
6. Be concise (2-3 sentences max)

Original text: {original_text}

Rewritten text:"""
            else:
                rewrite_prompt = f"""Rewrite this restaurant recommendation text to:
1. Match the actual {len(ranked)} restaurants shown: {', '.join(restaurant_names[:5])}
2. Only mention dishes suitable for {diet_label} diet (NO meat, fish, eggs, or animal products if vegetarian)
3. Keep the tone friendly and helpful
4. Be concise (2-3 sentences max)

Original text: {original_text}

Rewritten text:"""

            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": rewrite_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=200
            )

            rewritten_text = completion.choices[0].message.content.strip()
            ai_json["response"]["text"] = rewritten_text
            print(f"[DEBUG] Rewrote response text for {diet_label} diet")
    except Exception as e:
        print(f"[DEBUG] Failed to rewrite response text: {e}")

    return ai_json

