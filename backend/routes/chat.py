"""
Chat endpoint for restaurant recommendations.
"""
from fastapi import HTTPException
from typing import Dict, Any, Optional, Tuple
import json
import re

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
        r'\bveg(?:etarian)?\s+(?:food|dish|meal|option|restaurant)',
        r'\bonly\s+veg(?:etarian)?',
        r'\bpure\s+veg(?:etarian)?',
        r'\bvegetarian\s+only',
        r'\bno\s+(?:meat|non-veg|nonveg)',
        r'\bplant[-\s]?based',
    ]

    # Non-vegetarian indicators
    nonveg_patterns = [
        r'\b(?:i am|i\'m|looking for|want|need|prefer)\s+(?:a\s+)?non[-\s]?veg(?:etarian)?\s+(?:food|dish|meal|option)',
        r'\bnon[-\s]?veg(?:etarian)?\s+(?:food|dish|meal|option|restaurant)',
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
        r'(?:i want|i need|looking for|get me|find|where.*get|where.*find)\s+(.+?)(?:\s+(?:near|in|at|and)|$)',
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
            if dish and len(dish) > 2:
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
        # "I want X" or "show me X"
        r"^(?:i\s+want|i'd\s+like|i\s+need|give\s+me|show\s+me|find|get\s+me|looking\s+for)\s+(.+?)(?:\s+near|\s+in|\s+at|\s*$)",
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
            # Filter out generic food type queries
            generic_terms = ["food", "something", "anything", "restaurant", "place", "restaurants", "places"]
            if dish_name not in generic_terms and len(dish_name) > 2:
                return dish_name

    return None


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

    # Extract location from query (overrides request.location if not provided)
    query_location = extract_location_from_query(request.query)
    if query_location and not request.location:
        request.location = query_location
        print(f"[DEBUG] Location extracted from query: {query_location}")

    # Calculate user taste vector
    user_taste_vec = user_profile_to_taste_vector(dummy_profile) if dummy_profile else [0.0] * 6
    fav_text = ""
    if favorite_dishes:
        try:
            fav_text = " ".join([
                (d.name if hasattr(d, "name") else (d.get("name") if isinstance(d, dict) else str(d)))
                for d in favorite_dishes
            ])
        except Exception:
            fav_text = ""
    
    inferred_user = infer_taste_from_text_hybrid(fav_text, semantic=USE_SEMANTIC_INGREDIENT_TASTE)
    user_taste_vec = combine_vectors(user_taste_vec, inferred_user, secondary_weight=0.35)

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
                    "similarity": 0.5  # Neutral similarity since we're showing the full menu
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
        print(f"[DEBUG] Dish-specific query detected: '{dish_query}'")

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

                # Rank by rating and taste similarity
                from recommendations import dish_recommendations_for_restaurant
                ranked_restaurants = []
                for rest in restaurants_with_dish[:10]:  # Limit to top 10
                    # Get the matched dish
                    matched_dish = rest.get("dish", dish_query)

                    # Get additional recommended dishes for this restaurant
                    menu_items = rest["metadata"].get("menu_items", [])
                    additional_dishes = dish_recommendations_for_restaurant(
                        menu_items=menu_items,
                        user_taste_vec=user_taste_vec,
                        diet_type=diet_type,
                        top_n=4  # Get 4 additional recommendations
                    )

                    # Put the matched dish first, then additional recommendations
                    recommended_dishes = [
                        {"name": matched_dish, "similarity": 1.0}  # Perfect match
                    ]

                    # Add additional dishes if they're different from the matched dish
                    for dish in additional_dishes:
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
            initial_text = f"Sorry, I couldn't find '{dish_not_found_name}' in our restaurant database. However, based on your preferences and the taste profile of these dishes, here are some recommendations you might enjoy:"
        else:
            initial_text = f"Sorry, I couldn't find '{dish_not_found_name}' in our restaurant database. However, based on your preferences and the taste profile of this dish, here are some recommendations you might enjoy:"
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
        top_k = max(final_max_results, 10)
        print(f"[DEBUG] Querying Pinecone with top_k={top_k}")
        query_res = pc_index.query(vector=qvec, top_k=top_k, include_metadata=True, namespace="restaurants")
        matches = query_res.get("matches", []) if isinstance(query_res, dict) else getattr(query_res, "matches", [])
        print(f"[DEBUG] Pinecone returned {len(matches)} matches")

        # Filter and rank recommendations
        ranked = filter_and_rank_recommendations(
            matches=matches,
            user_taste_vec=user_taste_vec,
            favorite_dishes=favorite_dishes,
            diet_type=diet_type,
            allergies=allergies,
            max_results=final_max_results
        )

        print(f"[DEBUG] Total ranked restaurants: {len(ranked)}")
        print(f"[DEBUG] Returning top {len(ranked)} recommendations")
        
    except Exception as e:
        print(f"[DEBUG] Error in ranking: {e}")
        import traceback
        traceback.print_exc()
        ranked = []

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
                rewrite_prompt = f"""The user asked for '{dish_not_found_name}' but it's not available in our database.
Rewrite this text to:
1. Apologize that '{dish_not_found_name}' is not available
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

