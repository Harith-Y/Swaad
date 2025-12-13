"""
Taste vector analysis and similarity calculations.
"""
from typing import List, Dict, Optional
import csv
from pathlib import Path
from embeddings import embed_text, calculate_cosine_similarity, combine_vectors
from pinecone_client import query_pinecone
from config import TASTE_VECTOR_SIZE, USE_SEMANTIC_INGREDIENT_TASTE
from models import UserProfile


# Cache for taste inference
_taste_infer_cache: Dict[str, List[float]] = {}
_ingredient_flavor_map: Optional[Dict[str, Dict]] = None


def load_ingredient_flavor_map() -> Dict[str, Dict]:
    """Load ingredient flavor data from CSV."""
    global _ingredient_flavor_map
    
    if _ingredient_flavor_map is not None:
        return _ingredient_flavor_map
    
    csv_path = Path("ingredient-flavor.csv")
    if not csv_path.exists():
        print(f"[WARNING] {csv_path} not found")
        _ingredient_flavor_map = {}
        return _ingredient_flavor_map
    
    flavor_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ingredient = row.get("ingredient", "").strip().lower()
            if ingredient:
                flavor_map[ingredient] = {
                    "sweet": float(row.get("sweet", 0)),
                    "salty": float(row.get("salty", 0)),
                    "sour": float(row.get("sour", 0)),
                    "bitter": float(row.get("bitter", 0)),
                    "umami": float(row.get("umami", 0)),
                    "spicy": float(row.get("spicy", 0)),
                }
    
    _ingredient_flavor_map = flavor_map
    return flavor_map


def infer_taste_from_text(text: str) -> List[float]:
    """Infer taste vector from text using keyword matching."""
    if not text:
        return [0.0] * TASTE_VECTOR_SIZE
    
    if text in _taste_infer_cache:
        return _taste_infer_cache[text]
    
    flavor_map = load_ingredient_flavor_map()
    text_lower = text.lower()
    
    matched_flavors = []
    for ingredient, flavors in flavor_map.items():
        if ingredient in text_lower:
            matched_flavors.append([
                flavors["sweet"],
                flavors["salty"],
                flavors["sour"],
                flavors["bitter"],
                flavors["umami"],
                flavors["spicy"],
            ])
    
    if not matched_flavors:
        result = [0.0] * TASTE_VECTOR_SIZE
    else:
        result = [sum(f[i] for f in matched_flavors) / len(matched_flavors) for i in range(TASTE_VECTOR_SIZE)]
    
    _taste_infer_cache[text] = result
    return result


def infer_taste_from_text_semantic(text: str) -> List[float]:
    """Infer taste vector from text using semantic search in Pinecone."""
    if not text:
        return [0.0] * TASTE_VECTOR_SIZE
    
    if text in _taste_infer_cache:
        return _taste_infer_cache[text]
    
    try:
        query_vec = embed_text(text)
        matches = query_pinecone(
            query_vector=query_vec,
            top_k=5,
            filter_dict={"type": "ingredient"},
            include_metadata=True
        )
        
        if not matches:
            result = [0.0] * TASTE_VECTOR_SIZE
        else:
            taste_vectors = []
            for m in matches:
                meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
                taste_vectors.append([
                    float(meta.get("sweet", 0)),
                    float(meta.get("salty", 0)),
                    float(meta.get("sour", 0)),
                    float(meta.get("bitter", 0)),
                    float(meta.get("umami", 0)),
                    float(meta.get("spicy", 0)),
                ])
            
            result = [sum(tv[i] for tv in taste_vectors) / len(taste_vectors) for i in range(TASTE_VECTOR_SIZE)]
        
        _taste_infer_cache[text] = result
        return result
        
    except Exception as e:
        print(f"[ERROR] Semantic taste inference failed: {e}")
        return infer_taste_from_text(text)


def infer_taste_from_text_hybrid(text: str, semantic: bool = False) -> List[float]:
    """Infer taste vector using semantic or keyword-based approach."""
    if semantic and USE_SEMANTIC_INGREDIENT_TASTE:
        return infer_taste_from_text_semantic(text)
    return infer_taste_from_text(text)


def taste_similarity(user_vec: List[float], item_vec: List[float]) -> float:
    """Calculate taste similarity between user and item vectors."""
    return calculate_cosine_similarity(user_vec, item_vec)


def user_profile_to_taste_vector(user_profile: UserProfile) -> List[float]:
    """Convert user profile to taste vector."""
    if not user_profile or not user_profile.favorite_dishes:
        return [0.0] * TASTE_VECTOR_SIZE
    
    dish_texts = [d.name for d in user_profile.favorite_dishes if d.name]
    if not dish_texts:
        return [0.0] * TASTE_VECTOR_SIZE
    
    combined_text = " ".join(dish_texts)
    return infer_taste_from_text_hybrid(combined_text, semantic=USE_SEMANTIC_INGREDIENT_TASTE)


def favorites_boost(menu_items: List[str], favorite_dishes: List[Dict]) -> float:
    """Calculate boost score based on favorite dishes."""
    if not menu_items or not favorite_dishes:
        return 0.0
    
    menu_lower = [m.lower() for m in menu_items]
    fav_names = [d.get("name", "").lower() for d in favorite_dishes if d.get("name")]
    
    matches = sum(1 for fav in fav_names if any(fav in menu for menu in menu_lower))
    return matches * 0.1

