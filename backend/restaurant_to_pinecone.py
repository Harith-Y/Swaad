"""
Script to transform restaurant data and upload to Pinecone.

This script:
1. Takes raw restaurant data (from JSON or dict)
2. Transforms it to the Restaurant structure with taste vectors
3. Uses ingredient-flavor.csv to generate taste profiles
4. Uploads vectors to Pinecone for semantic search
"""

import os
import csv
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize models
_embedding_model = None
_ingredient_flavor_map = None


def get_embedding_model():
    """Get or initialize the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        print(f"Loading sentence-transformer model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def load_ingredient_flavor_map() -> Dict[str, Dict]:
    """Load ingredient-flavor.csv and create a mapping."""
    global _ingredient_flavor_map
    if _ingredient_flavor_map is not None:
        return _ingredient_flavor_map

    root = Path(__file__).resolve().parent.parent
    csv_path = root / "ingredient-flavor.csv"
    mapping: Dict[str, Dict] = {}

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found!")
        _ingredient_flavor_map = mapping
        return _ingredient_flavor_map

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ing = (row.get("ingredient") or "").strip()
            if not ing:
                continue
            spicy = float(row.get("spicy") or 0)
            sweet = float(row.get("sweet") or 0)
            umami = float(row.get("umami") or 0)
            sour = float(row.get("sour") or 0)
            salty = float(row.get("salty") or 0)
            bitter = 0.0
            taste_vector = [sweet, salty, sour, bitter, umami, spicy]
            mapping[ing.lower()] = {
                "ingredient": ing,
                "flavor_profile": {
                    "spicy": spicy,
                    "sweet": sweet,
                    "umami": umami,
                    "sour": sour,
                    "salty": salty
                },
                "taste_vector": taste_vector
            }

    _ingredient_flavor_map = mapping
    print(f"Loaded {len(mapping)} ingredients from ingredient-flavor.csv")
    return _ingredient_flavor_map


def infer_taste_from_text(text: str) -> List[float]:
    """
    Infer taste vector from text (dish names, menu items) using ingredient-flavor.csv.
    Returns: [sweet, salty, sour, bitter, umami, spicy]
    """
    m = load_ingredient_flavor_map()
    if not text:
        return [0.0] * 6
    t = text.lower()

    hits = 0
    acc = np.zeros(6, dtype=float)
    for ing_lc, info in m.items():
        if ing_lc and re.search(r"\b" + re.escape(ing_lc) + r"\b", t):
            vec = info.get("taste_vector") or [0.0] * 6
            if len(vec) == 6:
                acc += np.array(vec, dtype=float)
                hits += 1

    if hits == 0:
        return [0.0] * 6
    out = (acc / float(hits)).tolist()
    return [float(x) for x in out]


def calculate_restaurant_taste_vector(menu_items: List[str], popular_dishes: List[str]) -> List[float]:
    """
    Calculate overall taste vector for a restaurant based on menu items and popular dishes.
    Gives more weight to popular dishes.
    """
    all_text = " ".join(menu_items + popular_dishes)
    taste_vec = infer_taste_from_text(all_text)
    return taste_vec


def embed_text(text: str) -> List[float]:
    """Generate embedding vector for text."""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def price_to_range(price: Optional[str]) -> Optional[int]:
    """Convert price string ($, $$, $$$, $$$$) to integer 1-4."""
    if not price:
        return None
    if isinstance(price, str):
        return len(price.strip()) if price.strip() else None
    return None


def transform_restaurant(raw_data: Dict[str, Any], restaurant_id: int) -> Dict[str, Any]:
    """
    Transform raw restaurant data to the Restaurant structure.

    Expected raw_data format:
    {
        "name": str,
        "location": {"address": str, "city": str, "lat": float, "lng": float},
        "cuisine_types": List[str],
        "price": str,  # "$", "$$", etc.
        "menu_items": List[str],
        "popular_dishes": List[str],
        "avg_rating": float
    }
    """
    menu_items = raw_data.get("menu_items", [])
    popular_dishes = raw_data.get("popular_dishes", [])

    # Calculate taste vector
    taste_vector = calculate_restaurant_taste_vector(menu_items, popular_dishes)

    # Build location dict
    location = raw_data.get("location", {})
    if isinstance(location, str):
        location = {"address": location}

    restaurant = {
        "id": restaurant_id,
        "name": raw_data.get("name", ""),
        "location": {
            "lat": location.get("lat"),
            "lng": location.get("lng"),
            "address": location.get("address", ""),
            "city": location.get("city", "")
        },
        "cuisine_types": raw_data.get("cuisine_types", []),
        "price_range": price_to_range(raw_data.get("price")),
        "menu_items": menu_items,
        "taste_vector": taste_vector,
        "avg_rating": raw_data.get("avg_rating", 0.0),
        "popular_dishes": popular_dishes
    }

    return restaurant


def create_pinecone_vector(restaurant: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a Pinecone vector entry from a Restaurant object.

    Returns:
    {
        "id": str,
        "values": List[float],  # embedding vector
        "metadata": Dict
    }
    """
    # Create text for embedding
    embed_text_parts = [
        restaurant.get("name", ""),
        " ".join(restaurant.get("cuisine_types", [])),
        " ".join(restaurant.get("menu_items", [])[:20]),  # Limit to avoid too long text
        " ".join(restaurant.get("popular_dishes", []))
    ]
    embed_text_str = " ".join([p for p in embed_text_parts if p])

    # Generate embedding
    embedding_vector = embed_text(embed_text_str)

    # Prepare metadata
    location = restaurant.get("location", {})
    taste_vector = restaurant.get("taste_vector", [0.0] * 6)

    metadata = {
        "name": restaurant.get("name"),
        "avg_rating": float(restaurant.get("avg_rating", 0.0)),
        "city": location.get("city", ""),  # Extract city to top-level for filtering
        "cuisine_types": restaurant.get("cuisine_types", []),
        "location_json": json.dumps(location),
        "coordinates_json": json.dumps({
            "latitude": location.get("lat"),
            "longitude": location.get("lng")
        }),
        "menu_items": restaurant.get("menu_items", [])[:30],  # Limit for metadata size
        "popular_dishes": restaurant.get("popular_dishes", [])[:20],
        "taste_0": float(taste_vector[0]),
        "taste_1": float(taste_vector[1]),
        "taste_2": float(taste_vector[2]),
        "taste_3": float(taste_vector[3]),
        "taste_4": float(taste_vector[4]),
        "taste_5": float(taste_vector[5]),
    }
    
    # Add price_range only if it's not None (Pinecone doesn't accept null values)
    if restaurant.get("price_range") is not None:
        metadata["price_range"] = restaurant.get("price_range")

    vector_entry = {
        "id": f"restaurant:{restaurant['id']}",
        "values": embedding_vector,
        "metadata": metadata
    }

    return vector_entry


def upload_to_pinecone(vectors: List[Dict[str, Any]], batch_size: int = 100):
    """Upload vectors to Pinecone in batches."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "menu-buddy")

    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")

    print(f"Connecting to Pinecone index: {index_name}")
    
    # Validate vectors before upload
    valid_vectors = []
    skipped_count = 0
    
    for vec in vectors:
        # Check if vector has required 'values' field
        if not vec.get("values"):
            print(f"[WARNING] Skipping vector {vec.get('id', 'unknown')}: missing 'values' field")
            skipped_count += 1
            continue
        
        # Check if values is a valid list
        if not isinstance(vec["values"], list) or len(vec["values"]) == 0:
            print(f"[WARNING] Skipping vector {vec.get('id', 'unknown')}: invalid 'values' format")
            skipped_count += 1
            continue
        
        valid_vectors.append(vec)
    
    if skipped_count > 0:
        print(f"⚠️  Skipped {skipped_count} invalid vectors (missing embeddings)")
    
    if not valid_vectors:
        print("❌ No valid vectors to upload!")
        return None
    
    print(f"📤 Uploading {len(valid_vectors)} valid vectors...")
    
    pc = Pinecone(api_key=api_key)

    # Check if index exists, create if not
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}")
        from pinecone import ServerlessSpec
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created successfully")

    index = pc.Index(index_name)

    # Upload in batches
    total = len(valid_vectors)
    for i in range(0, total, batch_size):
        batch = valid_vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace="restaurants")  # Always use restaurants namespace
        print(f"Uploaded batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ({len(batch)} vectors)")

    print(f"✅ Successfully uploaded {total} vectors to Pinecone (namespace: restaurants)")
    
    if skipped_count > 0:
        print(f"⚠️  Note: {skipped_count} restaurants were skipped due to missing data")
    
    return index


def process_restaurants(restaurants_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a list of raw restaurant data and upload to Pinecone.

    Args:
        restaurants_data: List of raw restaurant dictionaries

    Returns:
        List of Pinecone vector entries
    """
    vectors = []

    for idx, raw_restaurant in enumerate(restaurants_data):
        restaurant_id = raw_restaurant.get("id", idx + 1)

        # Transform to Restaurant structure
        restaurant = transform_restaurant(raw_restaurant, restaurant_id)

        # Create Pinecone vector
        vector = create_pinecone_vector(restaurant)
        vectors.append(vector)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(restaurants_data)} restaurants")

    return vectors



def main():
    """
    Main function to demonstrate usage.

    Example usage:
    1. Load restaurant data from JSON file
    2. Process and upload to Pinecone
    """
    # Example restaurant data
    sample_restaurants = [
        {
            "id": 1,
            "name": "Spice Garden Indian Restaurant",
            "location": {
                "address": "123 Main St",
                "city": "San Francisco",
                "lat": 37.7749,
                "lng": -122.4194
            },
            "cuisine_types": ["Indian", "North Indian"],
            "price": "$$",
            "menu_items": [
                "Paneer Butter Masala",
                "Chicken Tikka Masala",
                "Garlic Naan",
                "Samosa",
                "Biryani",
                "Dal Makhani"
            ],
            "popular_dishes": [
                "Paneer Butter Masala",
                "Chicken Tikka Masala",
                "Garlic Naan"
            ],
            "avg_rating": 4.5
        },
        {
            "id": 2,
            "name": "Sushi Palace",
            "location": {
                "address": "456 Ocean Ave",
                "city": "San Francisco",
                "lat": 37.7849,
                "lng": -122.4094
            },
            "cuisine_types": ["Japanese", "Sushi"],
            "price": "$$$",
            "menu_items": [
                "California Roll",
                "Salmon Sashimi",
                "Tuna Nigiri",
                "Miso Soup",
                "Tempura",
                "Edamame"
            ],
            "popular_dishes": [
                "Salmon Sashimi",
                "California Roll"
            ],
            "avg_rating": 4.7
        }
    ]

    print("=" * 60)
    print("Restaurant to Pinecone Uploader")
    print("=" * 60)

    # Process restaurants
    print(f"\nProcessing {len(sample_restaurants)} restaurants...")
    vectors = process_restaurants(sample_restaurants)

    print(f"\nGenerated {len(vectors)} vectors")

    # Upload to Pinecone
    print("\nUploading to Pinecone...")
    upload_to_pinecone(vectors)

    print("\n" + "=" * 60)
    print("Upload complete!")
    print("=" * 60)


def load_from_json(json_file_path: str) -> List[Dict[str, Any]]:
    """Load restaurant data from a JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Load from JSON file if provided
        json_path = sys.argv[1]
        print(f"Loading restaurants from {json_path}...")
        restaurants = load_from_json(json_path)

        print(f"Processing {len(restaurants)} restaurants...")
        vectors = process_restaurants(restaurants)

        print(f"\nUploading {len(vectors)} vectors to Pinecone...")
        upload_to_pinecone(vectors)

        print("\nDone!")
    else:
        # Run with sample data
        main()


