"""
Restaurant and dish recommendation logic.
"""
from typing import List, Dict, Optional
import re
from embeddings import embed_text, calculate_cosine_similarity
from taste_analysis import taste_similarity, infer_taste_from_text_hybrid
from dish_processing import filter_dishes_by_diet, allergy_filter, filter_dishes_by_allergy
from config import USE_SEMANTIC_DISH_TASTE


def dish_recommendations_for_restaurant(
    menu_items,  # Can be List[str] or List[Dict] with pre-calculated taste vectors
    user_taste_vec: List[float],
    diet_type: Optional[str],
    allergies: List[str] = None,
    top_n: int = 5
) -> List[Dict]:
    """
    Get recommended dishes from a restaurant's menu based on user taste preferences.

    Args:
        menu_items: Either list of dish names (strings) or list of dicts with 'name' and 'taste' keys
        user_taste_vec: User's taste preference vector [sweet, salty, sour, bitter, umami, spicy]
        diet_type: Diet filter (veg, non-veg, mix)
        allergies: List of user allergies
        top_n: Number of top dishes to return
    """
    if not menu_items:
        return []

    # Check if menu_items has pre-calculated taste vectors
    has_taste_vectors = isinstance(menu_items, list) and len(menu_items) > 0 and isinstance(menu_items[0], dict)

    if has_taste_vectors:
        # Use pre-calculated taste vectors from Pinecone
        dishes = menu_items
    else:
        # Legacy: menu_items is just a list of strings
        dishes = [{"name": dish, "taste": None} for dish in menu_items]

    # Filter by diet
    dish_names = [d.get("name") if isinstance(d, dict) else d for d in dishes]
    filtered_names = filter_dishes_by_diet(dish_names, diet_type)
    
    # Filter by allergies
    if allergies:
        filtered_names = filter_dishes_by_allergy(filtered_names, allergies)

    if not filtered_names:
        return []

    # Calculate similarity for each dish
    dish_scores = []
    for dish in dishes:
        dish_name = dish.get("name") if isinstance(dish, dict) else dish
        
        if dish_name not in filtered_names:
            continue

        if dish_name not in filtered_names:
            continue

        # Get taste vector (pre-calculated or calculate on-the-fly)
        if has_taste_vectors and dish.get("taste"):
            dish_taste_vec = dish["taste"]
        else:
            dish_taste_vec = infer_taste_from_text_hybrid(dish_name, semantic=USE_SEMANTIC_DISH_TASTE)

        # Calculate similarity
        similarity = taste_similarity(user_taste_vec, dish_taste_vec)

        # Handle zero vectors
        user_sum = sum(abs(x) for x in user_taste_vec)
        dish_sum = sum(abs(x) for x in dish_taste_vec)
        if user_sum == 0 or dish_sum == 0:
            similarity = 0.5

        dish_scores.append({
            "name": dish_name,
            "similarity": round(similarity, 3)
        })

    # Sort by similarity and return top N
    dish_scores.sort(key=lambda x: x["similarity"], reverse=True)
    return dish_scores[:top_n]


def rank_restaurants(
    restaurants: List[Dict],
    user_taste_vec: List[float],
    favorite_dishes: List[Dict],
    query_embedding: Optional[List[float]] = None
) -> List[Dict]:
    """
    Rank restaurants based on taste similarity, favorites, and semantic match.
    """
    from taste_analysis import favorites_boost
    
    for restaurant in restaurants:
        # Get base score (from Pinecone semantic search)
        semantic_score = restaurant.get("score", 0.0)
        
        # Calculate taste similarity
        taste_vec = restaurant.get("taste_vector", [0.0] * 6)
        taste_score = taste_similarity(user_taste_vec, taste_vec)
        
        # Calculate favorites boost
        menu_items = restaurant.get("menu_items", [])
        fav_boost = favorites_boost(menu_items, favorite_dishes)
        
        # Combined score
        combined_score = semantic_score + 0.35 * taste_score + fav_boost
        restaurant["score"] = combined_score
    
    # Sort by combined score
    restaurants.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return restaurants


def calculate_restaurant_taste_vector(menu_items: List[str]) -> List[float]:
    """
    Calculate aggregate taste vector for a restaurant based on its menu.
    """
    if not menu_items:
        return [0.0] * 6
    
    taste_vectors = []
    for item in menu_items:
        taste_vec = infer_taste_from_text_hybrid(item, semantic=USE_SEMANTIC_DISH_TASTE)
        taste_vectors.append(taste_vec)
    
    if not taste_vectors:
        return [0.0] * 6
    
    # Average all taste vectors
    avg_taste = [
        sum(tv[i] for tv in taste_vectors) / len(taste_vectors)
        for i in range(6)
    ]
    
    return avg_taste


def filter_and_rank_recommendations(
    matches: List[Dict],
    user_taste_vec: List[float],
    favorite_dishes: List[Dict],
    diet_type: Optional[str],
    allergies: List[str],
    max_results: int = 10,
    query_text: Optional[str] = None,
    location_filter: Optional[str] = None
) -> List[Dict]:
    """
    Filter and rank restaurant recommendations from Pinecone matches.
    """
    from dish_processing import check_location_match
    
    # Pre-process query for keyword matching
    query_tokens = set()
    if query_text:
        # Simple tokenization: remove punctuation, lowercase
        q_clean = re.sub(r"[^\w\s]", "", query_text.lower())
        query_tokens = set(q_clean.split())
        # Remove common stop words
        stop_words = {"i", "want", "to", "eat", "some", "a", "the", "in", "at", "near", "me", "place", "restaurant", "find", "show", "give", "food", "good", "best", "delicious", "yummy", "looking", "for"}
        query_tokens = query_tokens - stop_words
    
    ranked = []
    
    for match in matches:
        # Extract metadata
        meta = match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", {})
        score = float(match.get("score", 0.0)) if isinstance(match, dict) else float(getattr(match, "score", 0.0))
        
        # Get menu items
        menu_items = meta.get("menu_items") or []
        
        # Filter by diet
        menu_items = filter_dishes_by_diet(menu_items, diet_type)
        if not menu_items:
            continue
        
        # Filter by allergies
        if allergies:
            menu_items = filter_dishes_by_allergy(menu_items, allergies)
            if not menu_items:
                continue
        
        # Parse location and coordinates
        location = meta.get("location")
        if location is None:
            loc_json = meta.get("location_json")
            if isinstance(loc_json, str) and loc_json:
                try:
                    import json
                    location = json.loads(loc_json)
                except Exception:
                    location = loc_json
        
        # Filter by location if provided
        if location_filter and location:
            loc_str = location
            if isinstance(location, dict):
                # Try to extract standard fields or join all values
                parts = []
                for key in ["address", "city", "state", "zip_code", "country"]:
                    if key in location and location[key]:
                        parts.append(str(location[key]))
                
                if parts:
                    loc_str = ", ".join(parts)
                else:
                    # Fallback: join all string values
                    loc_str = ", ".join([str(v) for v in location.values() if isinstance(v, (str, int))])
            
            # Ensure it's a string
            if not isinstance(loc_str, str):
                loc_str = str(loc_str)

            if not check_location_match(location_filter, loc_str):
                continue
        
        coordinates = meta.get("coordinates")
        if coordinates is None:
            coords_json = meta.get("coordinates_json")
            if isinstance(coords_json, str) and coords_json:
                try:
                    import json
                    coordinates = json.loads(coords_json)
                except Exception:
                    coordinates = coords_json
        
        # Get taste vector
        taste_vec = None
        try:
            t0 = meta.get("taste_0")
            t1 = meta.get("taste_1")
            t2 = meta.get("taste_2")
            t3 = meta.get("taste_3")
            t4 = meta.get("taste_4")
            t5 = meta.get("taste_5")
            if all(isinstance(x, (int, float)) for x in [t0, t1, t2, t3, t4, t5]):
                taste_vec = [float(t0), float(t1), float(t2), float(t3), float(t4), float(t5)]
        except Exception:
            taste_vec = None
        
        if not taste_vec:
            taste_vec = [0.0] * 6
        
        # Calculate scores
        from taste_analysis import favorites_boost
        tscore = taste_similarity(user_taste_vec, taste_vec)
        boost = favorites_boost(menu_items, favorite_dishes)
        
        # Calculate query relevance boost
        query_boost = 0.0
        if query_tokens:
            # Check menu items
            menu_text = " ".join(menu_items).lower()
            # Check popular dishes
            pop_dishes = meta.get("popular_dishes") or []
            if isinstance(pop_dishes, list):
                pop_text = " ".join([str(p) for p in pop_dishes]).lower()
            else:
                pop_text = ""
            
            # Check name and cuisine
            name_text = (meta.get("name") or "").lower()
            cuisine_text = " ".join(meta.get("cuisine_types") or []).lower()
            
            full_text = menu_text + " " + pop_text + " " + name_text + " " + cuisine_text
            full_text_clean = re.sub(r"[^\w\s]", "", full_text)
            restaurant_tokens = set(full_text_clean.split())
            
            overlap = len(query_tokens.intersection(restaurant_tokens))
            if overlap > 0:
                query_boost = 0.5 + (0.2 * overlap)

        combined = score + 0.35 * tscore + boost + query_boost
        
        # Get recommended dishes
        # Check if metadata has pre-calculated dish taste vectors (stored as JSON)
        dishes_with_taste = None
        dishes_json = meta.get("dishes_json")
        if dishes_json:
            try:
                import json
                dishes_with_taste = json.loads(dishes_json)
            except Exception as e:
                print(f"[WARNING] Failed to parse dishes_json: {e}")
                dishes_with_taste = None

        if dishes_with_taste:
            # Use pre-calculated taste vectors
            recommended_dishes = dish_recommendations_for_restaurant(
                dishes_with_taste, user_taste_vec, diet_type, top_n=5
            )
        else:
            # Fallback to calculating on-the-fly
            recommended_dishes = dish_recommendations_for_restaurant(
                menu_items, user_taste_vec, diet_type, top_n=5
            )
        
        # Build restaurant object
        ranked.append({
            "id": match.get("id") if isinstance(match, dict) else getattr(match, "id", None),
            "name": meta.get("name"),
            "url": meta.get("url"),
            "avg_rating": meta.get("avg_rating"),
            "price_range": meta.get("price_range"),
            "cuisine_types": meta.get("cuisine_types"),
            "location": location,
            "coordinates": coordinates,
            "menu_items": menu_items,
            "popular_dishes": meta.get("popular_dishes"),
            "taste_vector": taste_vec,
            "recommended_dishes": recommended_dishes,
            "photos": meta.get("photos"),
            "menu_url": meta.get("menu_url"),
            "score": combined
        })
    
    # Sort by score
    ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    # Limit results
    return ranked[:max_results]

