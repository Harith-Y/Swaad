"""
Chat endpoint for restaurant recommendations.
"""
from fastapi import HTTPException
from typing import Dict, Any, Optional, Tuple
import json
import re

from models import ChatRequest
from database import get_dummy_user, sync_dummy_user_from_request, dummy_user_to_user_profile
from embeddings import embed_text, combine_vectors
from taste_analysis import user_profile_to_taste_vector, infer_taste_from_text_hybrid
from pinecone_client import get_pinecone_index, maybe_upsert_ingredients_to_pinecone
from recommendations import filter_and_rank_recommendations
from dish_processing import get_groq_client
from config import GROQ_API_KEY, USE_SEMANTIC_INGREDIENT_TASTE


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

    Returns:
        Dish name if detected, None otherwise
    """
    query_lower = query.lower().strip()

    # Patterns for dish queries - more specific patterns
    patterns = [
        # "is there a place where X is available"
        r"is\s+there\s+(?:a\s+)?(?:place|restaurant)\s+(?:where|that\s+has)\s+(.+?)\s+(?:is\s+)?available",
        # "where can I find X" or "where can I get X"
        r"where\s+can\s+i\s+(?:find|get)\s+(.+?)(?:\s+near|\s+in|\s+at|\s*$)",
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

    # Check if this is a dish-specific query (not at a specific restaurant)
    dish_query = is_dish_query(request.query)
    if dish_query and "where restaurant is" not in request.query.lower():
        print(f"[DEBUG] Dish-specific query detected: '{dish_query}'")

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
                # Dish not found - continue to general recommendations below
                print(f"[DEBUG] Dish '{dish_query}' not found in any restaurant, will provide general recommendations")
                # Set a flag to modify the response text later
                dish_not_found = True
                dish_not_found_name = dish_query
            else:
                # Dish found - return restaurants that have it
                print(f"[DEBUG] Found '{dish_query}' at {len(restaurants_with_dish)} restaurants")

                # Rank by rating and taste similarity
                from recommendations import dish_recommendations_for_restaurant
                ranked_restaurants = []
                for rest in restaurants_with_dish[:10]:  # Limit to top 10
                    # Get recommended dishes for this restaurant
                    menu_items = rest["metadata"].get("menu_items", [])
                    recommended_dishes = dish_recommendations_for_restaurant(
                        menu_items=menu_items,
                        user_taste_vec=user_taste_vec,
                        diet_type=diet_type,
                        top_n=5
                    )

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
        initial_text = f"Sorry, I couldn't find '{dish_not_found_name}' in our restaurant database. But don't worry! Based on your taste preferences, here are some similar recommendations you might enjoy in {fallback_location}:"
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

