"""Check for restaurants that appear in multiple cities"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import json

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "menu-buddy"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Fetch all restaurants
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

query_vec = model.encode("restaurant").tolist()
results = index.query(
    vector=query_vec,
    top_k=10000,  # Get all (increased limit)
    include_metadata=True,
    namespace="restaurants"
)

matches = results.get("matches", [])
print(f"Total restaurants in Pinecone: {len(matches)}\n")

# Group by restaurant name
from collections import defaultdict
restaurants_by_name = defaultdict(list)

for m in matches:
    meta = m.get("metadata", {})
    name = meta.get("name", "Unknown")
    
    # Parse location
    location_json = meta.get("location_json", "{}")
    try:
        location = json.loads(location_json) if isinstance(location_json, str) else location_json
        city = location.get("city", "Unknown") if isinstance(location, dict) else "Unknown"
    except:
        city = "Unknown"
    
    restaurants_by_name[name].append(city)

# Find restaurants in multiple cities
multi_city = {name: cities for name, cities in restaurants_by_name.items() if len(set(cities)) > 1}

print("=" * 60)
print("RESTAURANTS IN MULTIPLE CITIES (Chains)")
print("=" * 60)

if multi_city:
    for name, cities in sorted(multi_city.items()):
        unique_cities = set(cities)
        print(f"\n{name}")
        print(f"  Locations: {', '.join(sorted(unique_cities))}")
        print(f"  Total: {len(cities)} locations")
else:
    print("\nNo restaurant chains found (all restaurants are single-location)")

print(f"\n📊 Summary:")
print(f"   Total unique restaurant names: {len(restaurants_by_name)}")
print(f"   Multi-city chains: {len(multi_city)}")
print(f"   Single-location: {len(restaurants_by_name) - len(multi_city)}")
