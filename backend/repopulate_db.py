import os
import asyncio
import json
import httpx
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add current directory to sys.path to import from main
sys.path.append(str(Path(__file__).parent))

# Import necessary functions from main.py
try:
    from main import (
        _menu_url_to_dishes,
        _embed_text,
        _get_pinecone_index,
        _infer_taste_from_text,
        _infer_taste_from_text_hybrid,
        _maybe_upsert_ingredients_to_pinecone,
        _load_ingredient_flavor_map,
        _combine_taste_vectors,
        _normalize_key,
        _get_embedding_model,
        _get_groq_client
    )
except ImportError as e:
    print(f"Error importing from main.py: {e}")
    sys.exit(1)

load_dotenv()

YELP_API_KEY = os.getenv("YELP_API_KEY")
YELP_V3_API_URL = "https://api.yelp.com/v3/businesses/search"

LOCATIONS = ["New York, NY", "San Francisco, CA"]
TARGET_PER_LOCATION = 50
FETCH_LIMIT = 60  # Fetch slightly more to account for missing menus

async def fetch_yelp_restaurants(location: str, limit: int = 50, offset: int = 0) -> List[Dict]:
    """
    Fetch restaurants from Yelp V3 API.
    """
    if not YELP_API_KEY:
        print("Error: YELP_API_KEY not found in environment variables.")
        return []

    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}",
        "Accept": "application/json"
    }
    
    params = {
        "location": location,
        "term": "restaurants",
        "limit": limit,
        "offset": offset,
        "sort_by": "best_match"
    }
    
    print(f"Fetching from Yelp V3 for {location} (limit={limit}, offset={offset})...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(YELP_V3_API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            businesses = data.get("businesses", [])
            print(f"Found {len(businesses)} businesses for {location}")
            return businesses
        except Exception as e:
            print(f"Error fetching from Yelp for {location}: {e}")
            return []

def calculate_menu_taste_vector(menu_items: List[str]) -> List[float]:
    """
    Calculate the average taste vector for a list of menu items.
    Uses hybrid approach (lexical + semantic) to handle cuisine types and unknown dishes.
    """
    if not menu_items:
        return [0.0] * 6
    
    vectors = []
    for item in menu_items:
        # Use hybrid inference to get better vectors for things like "Japanese" or "Pizza"
        # even if they aren't exact matches in the ingredient map
        vec = _infer_taste_from_text_hybrid(item, semantic=True)
        vectors.append(vec)
    
    if not vectors:
        return [0.0] * 6
    
    avg_vec = np.mean(vectors, axis=0).tolist()
    return [float(x) for x in avg_vec]

async def generate_menu_with_ai(restaurant_name: str, cuisine: str) -> List[str]:
    """
    Generate a plausible menu using Groq if scraping fails.
    """
    try:
        groq_client = _get_groq_client()
        if not groq_client:
            return []
            
        prompt = f"""Generate 15 realistic menu items for a restaurant named "{restaurant_name}" that serves {cuisine} cuisine.
        Return ONLY a JSON array of strings, e.g. ["Dish 1", "Dish 2"]. Do not include prices or descriptions."""
        
        def _call_groq():
            return groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.7
            )

        completion = await asyncio.to_thread(_call_groq)
        content = completion.choices[0].message.content
        data = json.loads(content)
        
        # Handle various JSON structures the LLM might return
        if isinstance(data, list):
            return [str(x) for x in data]
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, list):
                    return [str(x) for x in val]
        
        return []
    except Exception as e:
        print(f"Error generating menu for {restaurant_name}: {e}")
        return []

async def process_restaurant(restaurant: Dict) -> Optional[Dict]:
    """
    Process a single restaurant: extract menu, calculate vectors.
    """
    name = restaurant.get("name")
    rid = restaurant.get("id")
    
    # Extract Menu URL
    menu_url = None
    attributes = restaurant.get("attributes")
    if attributes and isinstance(attributes, dict):
        menu_url = attributes.get("MenuUrl")
    
    if not menu_url:
        # Try to find menu url in other fields if available, or fallback to main url
        # Note: Yelp V3 business object doesn't always have 'attributes' with MenuUrl.
        # It usually just has 'url'. We might need to scrape the Yelp page to find the menu link,
        # but for now let's use the provided 'url' if 'menu_url' is missing, 
        # though _menu_url_to_dishes might fail if it's just the Yelp page.
        # However, main.py logic often relies on 'menu_url' being present.
        # Let's try to use the 'url' field as a fallback, but _menu_url_to_dishes 
        # is designed to handle direct menu links or pages.
        menu_url = restaurant.get("url")

    if not menu_url:
        print(f"Skipping {name}: No URL found")
        return None
        
    print(f"Processing {name}...")
    
    # Extract menu items
    menu_items = []
    try:
        # We use the main.py function which handles fetching and parsing
        # It might be slow, so we rely on concurrency
        menu_items = await _menu_url_to_dishes(menu_url)
    except Exception as e:
        print(f"Error extracting menu for {name}: {e}")
        
    # If no menu items found from URL, try to use categories as fallback dishes
    if not menu_items:
        print(f"  - No menu items found for {name} via scraping.")
        # User explicitly requested NO AI generation.
        # We will try to use categories as a very basic fallback if scraping fails completely,
        # but we will NOT hallucinate dishes.
        categories = restaurant.get("categories", [])
        menu_items = [c.get("title") for c in categories if c.get("title")]
        
    if not menu_items:
        print(f"Skipping {name}: No menu items or categories found")
        return None
        
    # Clean menu items
    menu_items = [m for m in menu_items if len(m) > 3]
    
    # Calculate taste vector
    taste_vector = calculate_menu_taste_vector(menu_items)
    
    # Create embedding text
    cuisine_types = [c.get("title", "") for c in restaurant.get("categories", [])]
    cuisines_str = ", ".join(cuisine_types)
    
    # Rich text for embedding
    # Include name, cuisines, and top menu items
    menu_text = ", ".join(menu_items[:30])
    text_to_embed = f"{name} ({cuisines_str}). Menu: {menu_text}"
    
    embedding = _embed_text(text_to_embed)
    
    # Prepare data object
    processed_data = {
        "id": rid,
        "name": name,
        "url": restaurant.get("url"),
        "menu_url": menu_url,
        "image_url": restaurant.get("image_url"),
        "location": restaurant.get("location", {}),
        "coordinates": restaurant.get("coordinates", {}),
        "cuisine_types": cuisine_types,
        "price_range": len(restaurant.get("price", "$")) if restaurant.get("price") else 1,
        "rating": restaurant.get("rating", 0.0),
        "menu_items": menu_items,
        "popular_dishes": menu_items[:10], # Heuristic: assume first few are popular/representative
        "taste_vector": taste_vector,
        "embedding": embedding,
        "photos": [restaurant.get("image_url")] if restaurant.get("image_url") else []
    }
    
    return processed_data

async def repopulate_db():
    print("Starting database repopulation...")
    
    # Ensure models are loaded
    _load_ingredient_flavor_map()
    _get_embedding_model()
    
    # Ensure ingredients are in Pinecone for semantic taste inference
    print("Ensuring ingredients are indexed for semantic taste inference...")
    _maybe_upsert_ingredients_to_pinecone()
    
    index = _get_pinecone_index()
    if not index:
        print("Error: Could not connect to Pinecone.")
        return

    # 1. Clear existing data
    print("Clearing existing data in 'restaurants' namespace...")
    try:
        index.delete(delete_all=True, namespace="restaurants")
        print("Data cleared.")
    except Exception as e:
        print(f"Error clearing data: {e}")
        # Continue anyway, upsert will overwrite if IDs match, but we want a clean slate

    all_restaurants = []
    
    # Semaphore for concurrency
    # Increase to 10 for faster processing since we have more targets
    semaphore = asyncio.Semaphore(10)

    async def process_with_semaphore(business):
        async with semaphore:
            return await process_restaurant(business)

    for location in LOCATIONS:
        print(f"\n--- Fetching for {location} ---")
        location_restaurants = []
        offset = 0
        
        while len(location_restaurants) < TARGET_PER_LOCATION:
            # Yelp max limit is 50
            fetch_size = 50 
            
            businesses = await fetch_yelp_restaurants(location, limit=fetch_size, offset=offset)
            
            if not businesses:
                print("No more businesses found.")
                break
                
            print(f"Processing {len(businesses)} businesses concurrently...")
            tasks = [process_with_semaphore(b) for b in businesses]
            results = await asyncio.gather(*tasks)
            
            valid_results = [r for r in results if r is not None]
            print(f"Successfully processed {len(valid_results)} restaurants in this batch")
            
            # Add unique ones only (by id)
            existing_ids = {r["id"] for r in location_restaurants}
            for r in valid_results:
                if r["id"] not in existing_ids:
                    location_restaurants.append(r)
            
            print(f"Total valid for {location}: {len(location_restaurants)}/{TARGET_PER_LOCATION}")
            
            offset += fetch_size
            if offset >= 200: # Safety break
                print("Reached max offset limit.")
                break
        
        # Take up to TARGET_PER_LOCATION
        all_restaurants.extend(location_restaurants[:TARGET_PER_LOCATION])
    
    if not all_restaurants:
        print("No restaurants processed. Exiting.")
        return
        
    print(f"\nUpserting {len(all_restaurants)} restaurants to Pinecone...")
    
    vectors = []
    for r in all_restaurants:
        # Prepare metadata compatible with main.py
        
        # Serialize complex objects for metadata
        location_json = json.dumps(r["location"])
        coordinates_json = json.dumps(r["coordinates"])
        
        metadata = {
            "type": "restaurant",
            "name": r["name"],
            "url": r["url"],
            "menu_url": r["menu_url"],
            "avg_rating": float(r["rating"]),
            "price_range": int(r["price_range"]),
            "cuisine_types": r["cuisine_types"],
            "city": r["location"].get("city", ""),
            "location_json": location_json,
            "coordinates_json": coordinates_json,
            "menu_items": r["menu_items"][:40], # Limit to avoid metadata size limits
            "popular_dishes": r["popular_dishes"][:20],
            "photos": r["photos"],
            
            # Taste vector components
            "taste_0": float(r["taste_vector"][0]),
            "taste_1": float(r["taste_vector"][1]),
            "taste_2": float(r["taste_vector"][2]),
            "taste_3": float(r["taste_vector"][3]),
            "taste_4": float(r["taste_vector"][4]),
            "taste_5": float(r["taste_vector"][5]),
            
            # Add location_key for filtering
            "location_key": r["location"].get("city", "").lower() if r["location"].get("city") else ""
        }
        
        vectors.append({
            "id": str(r["id"]), # Ensure ID is string
            "values": r["embedding"],
            "metadata": metadata
        })
        
    # Batch upsert
    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch, namespace="restaurants")
            print(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")
        except Exception as e:
            print(f"Error upserting batch: {e}")

    print("Repopulation complete!")

if __name__ == "__main__":
    asyncio.run(repopulate_db())
