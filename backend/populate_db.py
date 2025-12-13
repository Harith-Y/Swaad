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
# We wrap this in try-except to handle potential import errors if main.py has runtime code
try:
    from main import (
        _menu_url_to_dishes,
        _embed_text,
        _get_pinecone_index,
        _infer_taste_from_text,
        _load_ingredient_flavor_map,
        _combine_taste_vectors,
        _normalize_key,
        _get_embedding_model
    )
except ImportError as e:
    print(f"Error importing from main.py: {e}")
    sys.exit(1)

load_dotenv()

YELP_API_KEY = os.getenv("YELP_API_KEY")
# Use the AI API endpoint as requested
YELP_AI_API_URL = "https://api.yelp.com/ai/chat/v2"

LOCATIONS = ["New York, NY", "San Francisco, CA"]
TARGET_PER_LOCATION = 10
FETCH_LIMIT = 20  # Fetch more to ensure we find ones with menus

async def fetch_yelp_restaurants(location: str, limit: int = 20) -> List[Dict]:
    """
    Fetch restaurants from Yelp AI API.
    """
    if not YELP_API_KEY:
        print("Error: YELP_API_KEY not found in environment variables.")
        return []

    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Construct the query
    query = f"List {limit} restaurants in {location}. Include menu url if available."
    
    payload = {
        "query": query,
        "request_context": {
            "skip_text_generation": True
        }
    }
    
    print(f"Fetching from Yelp for {location}...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(YELP_AI_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            print(f"DEBUG: Response Keys: {list(data.keys())}")
            if 'businesses' not in data and 'structured_data' not in data:
                 print(f"DEBUG: Response dump: {json.dumps(data)[:500]}")

            # Extract businesses from the response
            # Note: The exact structure of Yelp AI API response for 'skip_text_generation=True' 
            # needs to be handled dynamically. We look for 'businesses' key.
            businesses = []
            
            # Check for businesses in top level or nested
            if 'businesses' in data:
                businesses = data['businesses']
            elif 'structured_data' in data and 'businesses' in data['structured_data']:
                businesses = data['structured_data']['businesses']
            elif 'entities' in data and isinstance(data['entities'], list):
                for entity in data['entities']:
                    if 'businesses' in entity:
                        businesses.extend(entity['businesses'])
            else:
                # Fallback: try to find any list that looks like businesses
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and 'name' in value[0]:
                        businesses = value
                        break
            
            print(f"Found {len(businesses)} businesses for {location}")
            return businesses
            
        except Exception as e:
            print(f"Error fetching from Yelp for {location}: {e}")
            if 'response' in locals():
                print(f"Response content: {response.text[:500]}")
            return []

def calculate_menu_taste_vector(menu_items: List[str]) -> List[float]:
    """
    Calculate the average taste vector for a list of menu items.
    """
    if not menu_items:
        return [0.0] * 6
    
    vectors = []
    for item in menu_items:
        # Use the existing inference logic
        # We can use _infer_taste_from_text which uses the ingredient map
        vec = _infer_taste_from_text(item)
        vectors.append(vec)
    
    if not vectors:
        return [0.0] * 6
    
    # Calculate average
    avg_vec = np.mean(vectors, axis=0).tolist()
    return [float(x) for x in avg_vec]

async def process_restaurant(restaurant: Dict) -> Optional[Dict]:
    """
    Process a single restaurant: extract menu, calculate vectors.
    """
    name = restaurant.get("name")
    
    # Improved Menu URL extraction logic
    menu_url = None
    attributes = restaurant.get("attributes")
    if attributes and isinstance(attributes, dict):
        menu_url = attributes.get("MenuUrl")
        if not menu_url:
            menu_url = attributes.get("BusinessUrl")

    if not menu_url:
        menu_url = restaurant.get("menu_url") or restaurant.get("url")
    
    if not menu_url:
        print(f"Skipping {name}: No menu URL")
        return None
        
    print(f"Processing {name} with URL: {menu_url}...")
    
    # Extract menu items
    # _menu_url_to_dishes is async
    try:
        print(f"  - Fetching menu content for {name}...")
        menu_items = await _menu_url_to_dishes(menu_url)
        print(f"  - Extracted {len(menu_items)} items for {name}")
    except Exception as e:
        print(f"Error extracting menu for {name}: {e}")
        menu_items = []
        
    if not menu_items:
        print(f"Skipping {name}: No menu items found")
        return None
        
    # Clean menu items (remove prices, short strings, etc if needed)
    # The existing _menu_url_to_dishes does some cleaning but we can refine
    menu_items = [m for m in menu_items if len(m) > 3]
    
    if len(menu_items) < 3:
        print(f"Skipping {name}: Too few menu items")
        return None
        
    # Calculate taste vector (6-dim)
    taste_vector = calculate_menu_taste_vector(menu_items)
    
    # Create semantic embedding for the menu
    # We combine the restaurant name, cuisine, and top menu items
    cuisine_types = restaurant.get("categories", [])
    if isinstance(cuisine_types, list):
        cuisines = ", ".join([c.get("title", "") if isinstance(c, dict) else str(c) for c in cuisine_types])
    else:
        cuisines = str(cuisine_types)
        
    # Create a rich text representation for embedding
    # "Restaurant Name (Cuisine). Menu: Item 1, Item 2, ..."
    menu_text = ", ".join(menu_items[:20]) # Limit to top 20 for embedding to avoid truncation issues
    text_to_embed = f"{name} ({cuisines}). Menu: {menu_text}"
    
    embedding = _embed_text(text_to_embed)
    
    # Prepare the object for Pinecone
    # Schema:
    # Restaurant {
    #     id: int
    #     name: str
    #     location: {lat, lng, address, city}
    #     cuisine_types: List[str]  # ["Indian", "North Indian"]
    #     price_range: int  # 1-4 ($, $$, $$$, $$$$)
    #     menu_items: List[str]
    #     taste_vector: [6 floats]  # [sweet, salty, sour, bitter, umami, spicy]
    #     avg_rating: float
    #     popular_dishes: List[str]
    # }
    
    location_data = restaurant.get("location", {})
    if isinstance(location_data, dict):
        address = location_data.get("address1", "")
        city = location_data.get("city", "")
    else:
        address = str(location_data)
        city = ""
        
    coordinates = restaurant.get("coordinates", {})
    
    processed_data = {
        "id": restaurant.get("id"),
        "name": name,
        "location": {
            "lat": coordinates.get("latitude"),
            "lng": coordinates.get("longitude"),
            "address": address,
            "city": city
        },
        "cuisine_types": [c.get("title", "") if isinstance(c, dict) else str(c) for c in restaurant.get("categories", [])],
        "price_range": len(restaurant.get("price", "$")) if restaurant.get("price") else 1,
        "menu_items": menu_items,
        "taste_vector": taste_vector,
        "avg_rating": restaurant.get("rating", 0.0),
        "popular_dishes": menu_items[:5], # Placeholder
        "embedding": embedding
    }
    
    return processed_data

async def populate_db():
    # Ensure models are loaded
    _load_ingredient_flavor_map()
    _get_embedding_model()
    
    index = _get_pinecone_index()
    if not index:
        print("Error: Could not connect to Pinecone.")
        return

    all_restaurants = []
    
    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(5)

    async def process_with_semaphore(business):
        async with semaphore:
            return await process_restaurant(business)

    for location in LOCATIONS:
        print(f"\n--- Processing {location} ---")
        businesses = await fetch_yelp_restaurants(location, limit=FETCH_LIMIT)
        
        print(f"Processing {len(businesses)} businesses concurrently...")
        tasks = [process_with_semaphore(b) for b in businesses]
        results = await asyncio.gather(*tasks)
        
        valid_results = [r for r in results if r is not None]
        print(f"Successfully processed {len(valid_results)} restaurants for {location}")
        
        # Take up to TARGET_PER_LOCATION
        all_restaurants.extend(valid_results[:TARGET_PER_LOCATION])
    
    if not all_restaurants:
        print("No restaurants processed.")
        return
        
    print(f"\nUpserting {len(all_restaurants)} restaurants to Pinecone...")
    
    vectors = []
    for r in all_restaurants:
        # Flatten metadata for Pinecone (it handles nested JSON but flat is safer for some queries)
        # Actually Pinecone supports JSON metadata now.
        
        metadata = {
            "type": "restaurant",
            "name": r["name"],
            "city": r["location"]["city"],
            "location_key": _normalize_key(r["location"]["city"]),
            "address": r["location"]["address"],
            "cuisine_types": r["cuisine_types"], # List[str] is supported
            "price_range": r["price_range"],
            "avg_rating": r["avg_rating"],
            "menu_items": r["menu_items"][:20], # Limit size
            "taste_sweet": r["taste_vector"][0],
            "taste_salty": r["taste_vector"][1],
            "taste_sour": r["taste_vector"][2],
            "taste_bitter": r["taste_vector"][3],
            "taste_umami": r["taste_vector"][4],
            "taste_spicy": r["taste_vector"][5],
            # Add indexed taste keys for main.py compatibility
            "taste_0": r["taste_vector"][0],
            "taste_1": r["taste_vector"][1],
            "taste_2": r["taste_vector"][2],
            "taste_3": r["taste_vector"][3],
            "taste_4": r["taste_vector"][4],
            "taste_5": r["taste_vector"][5],
            # Store full JSON as string if needed for retrieval
            "full_data": json.dumps({k:v for k,v in r.items() if k != "embedding"})
        }
        
        vectors.append({
            "id": f"restaurant:{r['id']}",
            "values": r["embedding"],
            "metadata": metadata
        })
        
    # Batch upsert
    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch, namespace="restaurants")
            print(f"Upserted batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Error upserting batch: {e}")

    print("Population complete!")

if __name__ == "__main__":
    asyncio.run(populate_db())
