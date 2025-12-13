"""
Simple script to update dish taste vectors in Pinecone.
"""
import os
import sys
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from taste_analysis import infer_taste_from_text, load_ingredient_flavor_map

load_dotenv()

# Load ingredient map
print("Loading ingredient database...")
flavor_map = load_ingredient_flavor_map()
print(f"✅ Loaded {len(flavor_map)} ingredients\n")

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "menu-buddy"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# Fetch restaurants
print("Fetching restaurants...")
query_vec = [0.1] * 384
result = index.query(vector=query_vec, top_k=100, include_metadata=True, namespace="restaurants")
restaurants = result.get('matches', [])
print(f"✅ Found {len(restaurants)} restaurants\n")

# Process each restaurant
updated = 0
for i, restaurant in enumerate(restaurants, 1):
    try:
        rest_id = restaurant.get('id')
        meta = restaurant.get('metadata', {})
        name = meta.get('name', 'Unknown')
        menu = meta.get('menu_items', [])
        
        print(f"[{i}/100] {name} ({len(menu)} dishes)...", end=" ")
        
        if not menu:
            print("SKIP (no menu)")
            continue
        
        # Calculate taste for each dish
        dishes = []
        for dish in menu:
            if not isinstance(dish, str):
                continue
            taste = infer_taste_from_text(dish)
            dishes.append({"name": dish, "taste": [round(x, 3) for x in taste]})
        
        # Restaurant average taste
        if dishes:
            avg_taste = [round(sum(d["taste"][i] for d in dishes) / len(dishes), 3) for i in range(6)]
        else:
            avg_taste = [0.0] * 6
        
        # Update metadata
        meta['dishes_json'] = json.dumps(dishes, ensure_ascii=True)
        meta['taste_0'] = avg_taste[0]
        meta['taste_1'] = avg_taste[1]
        meta['taste_2'] = avg_taste[2]
        meta['taste_3'] = avg_taste[3]
        meta['taste_4'] = avg_taste[4]
        meta['taste_5'] = avg_taste[5]
        
        # Re-embed
        desc = f"{name} {' '.join(meta.get('cuisine_types', []))}"
        embedding = model.encode(desc).tolist()
        
        # Upsert
        index.upsert(vectors=[{"id": rest_id, "values": embedding, "metadata": meta}], namespace="restaurants")
        
        updated += 1
        print(f"✅ (taste: {avg_taste})")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        continue

print(f"\n✅ Updated {updated}/100 restaurants!")

