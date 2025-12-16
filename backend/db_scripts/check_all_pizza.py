"""
Comprehensive Pizza Restaurant Check

Checks for pizza restaurants across all locations in Pinecone
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("menu-buddy")

# Use semantic search for pizza
model = SentenceTransformer("all-MiniLM-L6-v2")
pizza_vec = model.encode("pizza restaurant").tolist()

print("🍕 Searching for pizza restaurants in Pinecone...")
print("="*80)

results = index.query(
    vector=pizza_vec,
    top_k=10000,
    namespace="restaurants",
    include_metadata=True
)

matches = results.matches

# Filter for restaurants with "pizza" in menu items or name
pizza_restaurants = []

for m in matches:
    meta = m.metadata
    name = meta.get('name', '').lower()
    menu_items = meta.get('menu_items', [])
    location_json = str(meta.get('location_json', ''))
    
    # Check if has pizza in name or menu
    has_pizza_name = 'pizza' in name
    has_pizza_menu = any('pizza' in str(item).lower() for item in menu_items)
    
    if has_pizza_name or has_pizza_menu:
        pizza_dishes = [item for item in menu_items if 'pizza' in str(item).lower()]
        
        # Parse location
        try:
            import json
            location = json.loads(location_json) if isinstance(location_json, str) else location_json
            city = location.get('city', 'Unknown') if isinstance(location, dict) else 'Unknown'
        except:
            city = 'Unknown'
        
        pizza_restaurants.append({
            'name': meta.get('name'),
            'city': city,
            'pizza_dishes': pizza_dishes[:5],  # First 5 pizza dishes
            'total_dishes': len(menu_items),
            'has_pizza_name': has_pizza_name,
            'has_pizza_menu': has_pizza_menu
        })

print(f"\n✅ Found {len(pizza_restaurants)} pizza restaurants!\n")

# Group by city
from collections import defaultdict
by_city = defaultdict(list)
for r in pizza_restaurants:
    by_city[r['city']].append(r)

# Show breakdown by city
print("📍 Pizza Restaurants by City:")
print("-"*80)
for city in sorted(by_city.keys()):
    count = len(by_city[city])
    print(f"\n{city}: {count} pizza restaurants")
    
    # Show first 3 from this city
    for i, rest in enumerate(by_city[city][:3], 1):
        print(f"  {i}. {rest['name']}")
        if rest['pizza_dishes']:
            print(f"     Pizza dishes: {', '.join(rest['pizza_dishes'][:3])}")
        else:
            print(f"     (Pizza in name, {rest['total_dishes']} total dishes)")

# Overall stats
print("\n" + "="*80)
print("📊 Summary:")
print(f"   Total pizza restaurants: {len(pizza_restaurants)}")
print(f"   With 'pizza' in name: {sum(1 for r in pizza_restaurants if r['has_pizza_name'])}")
print(f"   With pizza in menu: {sum(1 for r in pizza_restaurants if r['has_pizza_menu'])}")
print(f"   Cities covered: {len(by_city)}")

# NYC specific
nyc_cities = ['New York', 'Brooklyn', 'Manhattan', 'Queens', 'Bronx']
nyc_count = sum(len(by_city[city]) for city in nyc_cities if city in by_city)
print(f"\n🗽 NYC Metro (NY, Brooklyn, Manhattan): {nyc_count} pizza restaurants")

# Bay Area specific  
bay_cities = ['San Francisco', 'San Jose', 'Oakland', 'Berkeley']
bay_count = sum(len(by_city[city]) for city in bay_cities if city in by_city)
print(f"🌉 Bay Area (SF, SJ, Oakland, Berkeley): {bay_count} pizza restaurants")
