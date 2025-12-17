"""
Check Chicago restaurants in the JSON file
"""
import json

with open('master_restaurants_20251217_033916.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check for Chicago restaurants
chicago = []
for r in data:
    location = r.get('location', '')
    city = r.get('city', '')
    
    # Convert to string for checking
    location_str = str(location).lower() if location else ''
    city_str = str(city).lower() if city else ''
    
    if 'chicago' in location_str or 'chicago' in city_str:
        chicago.append(r)

print(f"Found {len(chicago)} Chicago restaurants in JSON file\n")

if chicago:
    print("Sample restaurant:")
    sample = chicago[0]
    print(f"  Name: {sample.get('name')}")
    print(f"  City: {sample.get('city')}")
    print(f"  Location: {sample.get('location')}")
    print(f"  Address: {sample.get('address')}")
    print(f"  Dishes: {len(sample.get('menu_items', []))}")
