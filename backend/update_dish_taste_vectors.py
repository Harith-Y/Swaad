"""
Script to update Pinecone restaurant data with dish-level taste vectors.
This calculates taste vectors based on ingredient-flavor.csv (1000 ingredients).
"""
import os
import sys
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from taste_analysis import infer_taste_from_text, load_ingredient_flavor_map
from tqdm import tqdm

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "menu-buddy")

print("=" * 60)
print("🍽️  UPDATING DISH TASTE VECTORS FROM INGREDIENT DATABASE")
print("=" * 60)

# Load ingredient flavor map
print("\n📊 Loading ingredient-flavor.csv...")
flavor_map = load_ingredient_flavor_map()
if not flavor_map:
    print("❌ ERROR: ingredient-flavor.csv not found or empty!")
    print("   Please ensure ingredient-flavor.csv is in the backend directory.")
    sys.exit(1)

print(f"✅ Loaded {len(flavor_map)} ingredients with flavor profiles")
print(f"   Sample ingredients: {list(flavor_map.keys())[:5]}")

# Initialize Pinecone
print("\n🔌 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"✅ Connected to index: {PINECONE_INDEX}")

# Get all restaurant vectors
print("\n🔍 Fetching all restaurants from Pinecone...")
query_vec = [0.1] * 384
result = index.query(
    vector=query_vec,
    top_k=100,
    include_metadata=True,
    namespace="restaurants"
)

restaurants = result.get('matches', [])
print(f"✅ Found {len(restaurants)} restaurants")

# Update each restaurant with dish taste vectors
print("\n🔄 Calculating taste vectors for all dishes...")
print("   Using ingredient-flavor.csv for accurate flavor profiles")
print()

updated_count = 0
total_dishes = 0

for restaurant in tqdm(restaurants, desc="Processing restaurants"):
    try:
        restaurant_id = restaurant.get('id')
        meta = restaurant.get('metadata', {})
        restaurant_name = meta.get('name', 'Unknown')
        menu_items = meta.get('menu_items', [])

        if not menu_items:
            continue

        # Calculate taste vector for each dish using ingredient database
        dishes_with_taste = []
        for dish_name in menu_items:
            # Ensure dish name is valid UTF-8
            if not isinstance(dish_name, str):
                continue

            # Use keyword-based matching with ingredient-flavor.csv
            taste_vec = infer_taste_from_text(dish_name)

            dishes_with_taste.append({
                "name": dish_name,
                "taste": [round(x, 3) for x in taste_vec]  # Round to 3 decimals
            })
            total_dishes += 1

        # Calculate restaurant-level taste vector (average of all dishes)
        if dishes_with_taste:
            restaurant_taste = [
                round(sum(d["taste"][i] for d in dishes_with_taste) / len(dishes_with_taste), 3)
                for i in range(6)
            ]
        else:
            restaurant_taste = [0.0] * 6

        # Update metadata
        # Store dishes as JSON string (Pinecone doesn't support nested objects)
        meta['dishes_json'] = json.dumps(dishes_with_taste, ensure_ascii=True)

        # Store taste vector components separately (Pinecone limitation)
        meta['taste_0'] = restaurant_taste[0]  # sweet
        meta['taste_1'] = restaurant_taste[1]  # salty
        meta['taste_2'] = restaurant_taste[2]  # sour
        meta['taste_3'] = restaurant_taste[3]  # bitter
        meta['taste_4'] = restaurant_taste[4]  # umami
        meta['taste_5'] = restaurant_taste[5]  # spicy

        # Get the original embedding vector
        restaurant_desc = f"{meta.get('name', '')} {' '.join(meta.get('cuisine_types', []))}"
        embedding = model.encode(restaurant_desc).tolist()

        # Upsert back to Pinecone
        index.upsert(
            vectors=[{
                "id": restaurant_id,
                "values": embedding,
                "metadata": meta
            }],
            namespace="restaurants"
        )

        updated_count += 1
    except Exception as e:
        print(f"\n[ERROR] Failed to update {restaurant_name}: {e}")
        continue

print()
print("=" * 60)
print("✅ UPDATE COMPLETE!")
print("=" * 60)
print(f"📊 Statistics:")
print(f"   - Restaurants updated: {updated_count}")
print(f"   - Total dishes processed: {total_dishes}")
print(f"   - Avg dishes per restaurant: {total_dishes / updated_count if updated_count > 0 else 0:.1f}")
print()
print("📝 Updated structure:")
print("   Each restaurant now has:")
print("   - 'dishes': [{'name': 'Pad Thai', 'taste': [0.2, 0.25, 0.0, 0.0, 0.45, 0.0]}, ...]")
print("   - 'taste_vector': [0.2, 0.3, ...] (restaurant average)")
print()
print("🎯 Taste vector format: [sweet, salty, sour, bitter, umami, spicy]")
print()
print("✨ Your recommendations will now be based on ingredient flavor profiles!")

