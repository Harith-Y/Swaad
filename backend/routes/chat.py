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
from dish_processing import get_groq_client, extract_dish_from_query, is_relevant_query, detect_diet_from_query, extract_location_from_query, classify_intent, extract_dish_and_restaurant, check_location_match
from config import GROQ_API_KEY, USE_SEMANTIC_INGREDIENT_TASTE





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

    # Override diet type if detected in query
    query_diet = detect_diet_from_query(request.query)
    if query_diet:
        print(f"[DEBUG] Detected diet from query: {query_diet} (overriding {diet_type})")
        diet_type = query_diet

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

    # Extract location from query (highest priority)
    query_location = extract_location_from_query(request.query)
    if query_location:
        print(f"[DEBUG] Extracted location from query: {query_location}")
        request.location = query_location

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

    # CRITICAL: If no location is found anywhere, ask the user for it
    if not fallback_location or not fallback_location.strip():
        print("[DEBUG] No location found in query or profile. Asking user for location.")
        # Save the current query as pending so we can process it after they provide location
        if isinstance(dummy_user, dict):
            dummy_user["pending_query"] = request.query
        
        return {
            "response": {
                "text": "I'd love to help you find that! 🌍 Could you please tell me which city or area you're in so I can find the best restaurants near you?"
            },
            "chat_id": request.chat_id,
            "menu_buddy": {"recommendations": []}
        }

    # Set max results
    final_max_results = request.max_results or 10
    print(f"[DEBUG] max_results set to: {final_max_results}")

    # Initialize flags
    dish_not_found = False
    dish_not_found_name = None

    # Classify intent
    intent = classify_intent(request.query)
    print(f"[DEBUG] Query intent: {intent}")

    if intent == "greeting":
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

    if intent == "irrelevant":
        print(f"[DEBUG] Irrelevant query detected: {request.query}")
        return {
            "response": {
                "text": "I apologize, but this is beyond my capability. I can only assist you with food and restaurant recommendations."
            },
            "chat_id": request.chat_id,
            "menu_buddy": {"recommendations": []}
        }

    # Check if this is a dish-specific query (not at a specific restaurant)
    # Use Groq for smarter extraction instead of regex
    dish_query = extract_dish_from_query(request.query)
    
    if dish_query:
        print(f"[DEBUG] Dish-specific query detected: '{dish_query}'")
        
        # If diet is veg, and dish query contains "veg", try to broaden search to base dish
        # e.g. "veg burger" -> search for "burger" but apply strict veg filter later
        search_query = dish_query
        if diet_type in ["veg", "vegetarian"] and "veg" in dish_query.lower():
            # Remove "veg", "vegetarian", "pure veg" from search query
            base_dish = re.sub(r"\b(veg|vegetarian|pure veg)\b", "", dish_query, flags=re.IGNORECASE).strip()
            if base_dish and len(base_dish) > 2:
                print(f"[DEBUG] Broadening search from '{dish_query}' to '{base_dish}' with strict veg filter")
                search_query = base_dish

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
                
                # HARD FILTER: Check if dish exists in menu (fuzzy match)
                has_dish = False
                matched_dish_name = ""
                
                # 1. Check exact substring match
                for menu_item in menu_items:
                    if search_query.lower() in menu_item.lower():
                        # If we broadened the search, we MUST check if the found item is veg
                        if search_query != dish_query and diet_type in ["veg", "vegetarian"]:
                             from dish_processing import is_nonveg_text
                             if is_nonveg_text(menu_item):
                                 continue
                        
                        has_dish = True
                        matched_dish_name = menu_item
                        break
                
                # 2. If no substring match, check word overlap (e.g. "burger" matches "Cheeseburger")
                if not has_dish:
                    dish_words = set(search_query.lower().split())
                    for menu_item in menu_items:
                        menu_words = set(re.sub(r"[^\w\s]", "", menu_item.lower()).split())
                        if dish_words.issubset(menu_words):
                            # If we broadened the search, we MUST check if the found item is veg
                            if search_query != dish_query and diet_type in ["veg", "vegetarian"]:
                                 from dish_processing import is_nonveg_text
                                 if is_nonveg_text(menu_item):
                                     continue

                            has_dish = True
                            matched_dish_name = menu_item
                            break
                if not has_dish:
                    dish_words = set(dish_query.lower().split())
                    for menu_item in menu_items:
                        menu_words = set(re.sub(r"[^\w\s]", "", menu_item.lower()).split())
                        if dish_words.issubset(menu_words):
                            has_dish = True
                            matched_dish_name = menu_item
                            break

                if has_dish:
                    # Check location filter if present
                    if fallback_location:
                        rest_location = meta.get("location")
                        if not rest_location:
                            loc_json = meta.get("location_json")
                            if isinstance(loc_json, str) and loc_json:
                                try:
                                    import json
                                    rest_location = json.loads(loc_json)
                                except Exception:
                                    rest_location = loc_json
                        
                        # Handle dict location
                        if isinstance(rest_location, dict):
                            parts = []
                            for key in ["address", "city", "state", "zip_code", "country"]:
                                if key in rest_location and rest_location[key]:
                                    parts.append(str(rest_location[key]))
                            if parts:
                                rest_location = ", ".join(parts)
                            else:
                                rest_location = ", ".join([str(v) for v in rest_location.values() if isinstance(v, (str, int))])
                        
                        if not rest_location: rest_location = ""
                        
                        if not check_location_match(fallback_location, str(rest_location)):
                             continue

                    restaurants_with_dish.append({
                        "name": meta.get("name"),
                        "rating": meta.get("avg_rating"),
                        "price_range": meta.get("price_range"),
                        "cuisine_types": meta.get("cuisine_types", []),
                        "dish": matched_dish_name,
                        "metadata": meta
                    })

            if not restaurants_with_dish:
                # Dish not found - continue to general recommendations below
                print(f"[DEBUG] Dish '{search_query}' not found in any restaurant, will provide general recommendations")
                # Set a flag to modify the response text later
                dish_not_found = True
                dish_not_found_name = dish_query
            else:
                # Dish found - return restaurants that have it
                print(f"[DEBUG] Found '{search_query}' at {len(restaurants_with_dish)} restaurants")

                # Rank by rating and taste similarity
                from recommendations import dish_recommendations_for_restaurant
                ranked_restaurants = []
                for rest in restaurants_with_dish[:10]:  # Limit to top 10
                    # Get recommended dishes for this restaurant
                    menu_items = rest["metadata"].get("menu_items", [])
                    
                    # Prioritize the requested dish in recommendations
                    recommended_dishes = dish_recommendations_for_restaurant(
                        menu_items=menu_items,
                        user_taste_vec=user_taste_vec,
                        diet_type=diet_type,
                        allergies=allergies,
                        top_n=5
                    )
                    
                    # Ensure the matched dish is at the top if it fits the diet AND allergies
                    matched_dish = rest["dish"]
                    from dish_processing import allergy_filter
                    if allergy_filter(matched_dish, allergies):
                        # Check if matched dish is already in recommendations
                        if not any(d["name"] == matched_dish for d in recommended_dishes):
                            # Add it to the top (with a high similarity score)
                            recommended_dishes.insert(0, {"name": matched_dish, "similarity": 0.99})
                            recommended_dishes = recommended_dishes[:5]
                    else:
                        # If the matched dish is allergic, we shouldn't recommend it even if it matched the search
                        # But we can still recommend the restaurant if it has other safe dishes
                        pass

                    if not recommended_dishes:
                        continue

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
    specific_query = extract_dish_and_restaurant(request.query)
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
            max_results=final_max_results,
            query_text=request.query,
            location_filter=fallback_location
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

