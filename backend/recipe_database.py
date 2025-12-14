"""
Recipe Database Module
Handles loading and searching the recipes_with_flavour_profiles.csv database
"""

import csv
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher

# Global cache for recipes
_recipes_cache: Optional[Dict[str, Dict[str, Any]]] = None
_recipes_list: Optional[List[Dict[str, Any]]] = None


def load_recipes_database() -> None:
    """
    Load recipes from CSV into memory for fast lookups.
    This is called once on startup.
    """
    global _recipes_cache, _recipes_list

    if _recipes_cache is not None:
        return  # Already loaded

    # Try multiple possible paths
    possible_paths = [
        Path("recipes_with_flavour_profiles.csv"),
        Path("../recipes_with_flavour_profiles.csv"),
        Path(__file__).parent.parent / "recipes_with_flavour_profiles.csv"
    ]

    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        print(f"[WARNING] recipes_with_flavour_profiles.csv not found in any location, recipe database will not be available")
        _recipes_cache = {}
        _recipes_list = []
        return
    
    print(f"[INFO] Loading recipe database from {csv_path}...")
    
    _recipes_cache = {}
    _recipes_list = []
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                recipe_name = row.get("name", "").strip().lower()
                if not recipe_name:
                    continue
                
                # Parse flavor profile
                flavor_str = row.get("flavor_profile", "{}")
                try:
                    # Replace single quotes with double quotes for JSON parsing
                    flavor_str = flavor_str.replace("'", '"')
                    flavor_profile = json.loads(flavor_str)
                except json.JSONDecodeError:
                    flavor_profile = {
                        "sweet": 0, "salty": 0, "sour": 0,
                        "bitter": 0, "umami": 0, "spicy": 0
                    }
                
                # Parse ingredients
                ingredients_str = row.get("ingredients", "[]")
                try:
                    ingredients_str = ingredients_str.replace("'", '"')
                    ingredients = json.loads(ingredients_str)
                except json.JSONDecodeError:
                    ingredients = []
                
                recipe_data = {
                    "id": row.get("id", ""),
                    "name": recipe_name,
                    "original_name": row.get("name", "").strip(),
                    "ingredients": ingredients,
                    "flavor_profile": flavor_profile
                }
                
                # Store in cache (use lowercase name as key)
                _recipes_cache[recipe_name] = recipe_data
                _recipes_list.append(recipe_data)
        
        print(f"[INFO] Loaded {len(_recipes_cache)} recipes into memory")
        
    except Exception as e:
        print(f"[ERROR] Failed to load recipe database: {e}")
        _recipes_cache = {}
        _recipes_list = []


def similarity_score(str1: str, str2: str) -> float:
    """Calculate similarity score between two strings (0-1)."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def search_recipe_by_name(dish_name: str, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
    """
    Search for a recipe by name in the database.
    
    Args:
        dish_name: Name of the dish to search for
        threshold: Minimum similarity score (0-1) to consider a match
    
    Returns:
        Recipe data if found, None otherwise
    """
    global _recipes_cache, _recipes_list
    
    # Ensure database is loaded
    if _recipes_cache is None:
        load_recipes_database()
    
    if not _recipes_cache:
        return None
    
    dish_name_lower = dish_name.lower().strip()
    
    # Try exact match first
    if dish_name_lower in _recipes_cache:
        recipe = _recipes_cache[dish_name_lower]
        print(f"[DEBUG] Found exact match for '{dish_name}' in recipe database")
        return recipe
    
    # Try fuzzy matching
    best_match = None
    best_score = threshold
    
    for recipe in _recipes_list[:10000]:  # Limit search to first 10K for performance
        score = similarity_score(dish_name_lower, recipe["name"])
        if score > best_score:
            best_score = score
            best_match = recipe
    
    if best_match:
        print(f"[DEBUG] Found fuzzy match for '{dish_name}': '{best_match['original_name']}' (score: {best_score:.2f})")
        return best_match
    
    print(f"[DEBUG] No match found for '{dish_name}' in recipe database")
    return None


def get_taste_vector_from_recipe(recipe: Dict[str, Any]) -> List[float]:
    """
    Extract taste vector from recipe data.
    
    Returns:
        List of 6 floats: [sweet, salty, sour, bitter, umami, spicy]
    """
    flavor = recipe.get("flavor_profile", {})
    
    taste_vector = [
        float(flavor.get("sweet", 0)),
        float(flavor.get("salty", 0)),
        float(flavor.get("sour", 0)),
        float(flavor.get("bitter", 0)),
        float(flavor.get("umami", 0)),
        float(flavor.get("spicy", 0))
    ]
    
    return taste_vector


def has_valid_taste_profile(recipe: Dict[str, Any]) -> bool:
    """Check if recipe has a non-zero taste profile."""
    taste_vector = get_taste_vector_from_recipe(recipe)
    return any(v > 0 for v in taste_vector)

