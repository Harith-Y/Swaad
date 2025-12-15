import json

# Load scraped restaurants
with open('scraped_restaurants.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find restaurants with "pizza" in menu
pizza_restaurants = []
for r in data:
    menu_items = r.get('menu_items', [])
    if menu_items:
        pizza_items = [item for item in menu_items if 'pizza' in item.lower()]
        if pizza_items:
            pizza_restaurants.append({
                'name': r['name'],
                'location': r.get('location', {}).get('city', 'Unknown'),
                'pizza_items': pizza_items[:5]
            })

print(f"Total restaurants in file: {len(data)}")
print(f"Restaurants with 'pizza' in menu: {len(pizza_restaurants)}")
print("\nFirst 10 restaurants with pizza:")
for i, r in enumerate(pizza_restaurants[:10], 1):
    print(f"\n{i}. {r['name']} ({r['location']})")
    print(f"   Pizza dishes: {', '.join(r['pizza_items'])}")
