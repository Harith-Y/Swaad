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
    ai_json = {
        "response": {
            "text": f"Here are some great restaurant recommendations for you in {fallback_location}!"
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

