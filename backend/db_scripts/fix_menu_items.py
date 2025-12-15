"""
Script to diagnose and fix menu_items vs cuisine_types issue in Pinecone.

This script:
1. Analyzes all restaurants in Pinecone
2. Identifies restaurants where menu_items are actually cuisine_types
3. Provides detailed report
4. Optionally removes problematic data so you can re-scrape properly
"""
import os
import sys
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from collections import Counter

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "menu-buddy")

# Common cuisine types that should NOT be in menu_items
COMMON_CUISINES = {
    'pizza', 'italian', 'thai', 'chinese', 'japanese', 'indian', 'mexican',
    'korean', 'vietnamese', 'american', 'french', 'greek', 'mediterranean',
    'spanish', 'middle eastern', 'sushi', 'pasta shops', 'burgers', 'sandwiches',
    'breakfast & brunch', 'brunch', 'salad', 'seafood', 'steakhouses',
    'vegetarian', 'vegan', 'gluten-free', 'bakeries', 'cafes', 'coffee & tea',
    'desserts', 'ice cream & frozen yogurt', 'juice bars & smoothies',
    'beer bar', 'wine bars', 'cocktail bars', 'pubs', 'sports bars',
    'food trucks', 'food stands', 'food court', 'buffets', 'diners',
    'fast food', 'bbq', 'bagels', 'delis', 'soup', 'noodles', 'dim sum',
    'tapas/small plates', 'gastropubs', 'bistros', 'brasseries', 'trattorias',
    'ramen', 'hot pot', 'fondue', 'comfort food', 'soul food', 'southern',
    'tex-mex', 'cajun/creole', 'caribbean', 'latin american', 'asian fusion',
    'fusion', 'farm-to-table', 'organic', 'halal', 'kosher', 'tapas'
}

print("=" * 70)
print("🔍  PINECONE MENU DATA DIAGNOSTIC TOOL")
print("=" * 70)

# Initialize Pinecone
print("\n🔌 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
print(f"✅ Connected to index: {PINECONE_INDEX}")

# Get all restaurant vectors
print("\n📥 Fetching all restaurants from Pinecone...")
query_vec = [0.1] * 384
result = index.query(
    vector=query_vec,
    top_k=10000,  # Get all restaurants
    include_metadata=True,
    namespace="restaurants"
)

restaurants = result.get('matches', [])
print(f"✅ Found {len(restaurants)} restaurants\n")

# Analyze data
print("=" * 70)
print("📊 ANALYSIS RESULTS")
print("=" * 70)

problematic_restaurants = []
good_restaurants = []
all_menu_items = []
cuisine_in_menu_count = 0

for restaurant in restaurants:
    meta = restaurant.get('metadata', {})
    restaurant_id = restaurant.get('id')
    restaurant_name = meta.get('name', 'Unknown')
    menu_items = meta.get('menu_items', [])
    cuisine_types = meta.get('cuisine_types', [])
    
    if not menu_items:
        continue
    
    # Check if menu items are actually cuisine types
    problematic_items = []
    valid_items = []
    
    for item in menu_items:
        item_lower = item.lower().strip()
        all_menu_items.append(item_lower)
        
        # Check if this looks like a cuisine type
        if item_lower in COMMON_CUISINES:
            problematic_items.append(item)
            cuisine_in_menu_count += 1
        else:
            valid_items.append(item)
    
    if problematic_items:
        problematic_restaurants.append({
            'id': restaurant_id,
            'name': restaurant_name,
            'menu_items': menu_items,
            'cuisine_types': cuisine_types,
            'problematic_items': problematic_items,
            'valid_items': valid_items
        })
    else:
        good_restaurants.append({
            'id': restaurant_id,
            'name': restaurant_name,
            'menu_count': len(menu_items)
        })

# Print summary
print(f"\n📈 SUMMARY:")
print(f"   Total Restaurants: {len(restaurants)}")
print(f"   ✅ Good Data: {len(good_restaurants)} ({len(good_restaurants)/len(restaurants)*100:.1f}%)")
print(f"   ❌ Problematic Data: {len(problematic_restaurants)} ({len(problematic_restaurants)/len(restaurants)*100:.1f}%)")
print(f"   🚨 Cuisine types in menu_items: {cuisine_in_menu_count} occurrences")

# Show most common "menu items" (likely cuisines)
print(f"\n🔝 TOP 20 MOST COMMON 'MENU ITEMS':")
item_counts = Counter(all_menu_items)
for item, count in item_counts.most_common(20):
    is_cuisine = "🚨 CUISINE" if item in COMMON_CUISINES else "✅ DISH"
    print(f"   {is_cuisine:12} | {item:30} ({count} restaurants)")

# Show sample problematic restaurants
print(f"\n" + "=" * 70)
print("❌ PROBLEMATIC RESTAURANTS (showing first 10)")
print("=" * 70)

for i, rest in enumerate(problematic_restaurants[:10], 1):
    print(f"\n{i}. {rest['name']} (ID: {rest['id']})")
    print(f"   Cuisine Types: {rest['cuisine_types']}")
    print(f"   Menu Items (WRONG): {rest['menu_items'][:10]}")
    print(f"   🚨 Problematic: {rest['problematic_items']}")
    if rest['valid_items']:
        print(f"   ✅ Valid: {rest['valid_items']}")

# Show sample good restaurants
print(f"\n" + "=" * 70)
print("✅ GOOD RESTAURANTS (showing first 5)")
print("=" * 70)

for i, rest in enumerate(good_restaurants[:5], 1):
    print(f"{i}. {rest['name']} - {rest['menu_count']} menu items")

# Provide recommendations
print(f"\n" + "=" * 70)
print("💡 RECOMMENDATIONS")
print("=" * 70)

if len(problematic_restaurants) > len(restaurants) * 0.5:
    print("\n⚠️  CRITICAL: More than 50% of your data is problematic!")
    print("\n   ROOT CAUSE:")
    print("   Your scraping script is storing cuisine_types in the menu_items field.")
    print("   Restaurant metadata should have:")
    print("     - cuisine_types: ['Italian', 'Pizza'] (categories)")
    print("     - menu_items: ['Margherita Pizza', 'Caesar Salad'] (actual dishes)")
    
    print("\n   SOLUTION:")
    print("   1. Fix your web scraper to properly extract actual dish names")
    print("   2. Clear the Pinecone index")
    print("   3. Re-scrape and re-upload with correct data")
    
    print("\n   QUICK FIX OPTIONS:")
    print("   a) Delete all problematic restaurants and re-scrape them")
    print("   b) Keep good restaurants, only re-scrape problematic ones")
    print("   c) Clear entire index and start fresh")
    
elif problematic_restaurants:
    print("\n⚠️  Some restaurants have incorrect data (cuisine types as menu items)")
    print(f"   {len(problematic_restaurants)} restaurants need to be re-scraped")
    print("\n   You can:")
    print("   1. Delete these specific restaurants")
    print("   2. Re-scrape them with fixed scraper")
    print("   3. Re-upload with correct menu_items")
else:
    print("\n✅ All restaurants have valid menu items!")
    print("   No action needed.")

# Offer to clean up
print(f"\n" + "=" * 70)
print("🧹 CLEANUP OPTIONS")
print("=" * 70)

if problematic_restaurants:
    print("\nWhat would you like to do?")
    print("1. Delete ONLY problematic restaurants (keep good ones)")
    print("2. Delete ALL restaurants (fresh start)")
    print("3. Export problematic restaurant IDs to file (for manual review)")
    print("4. Exit (no changes)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        confirm = input(f"\n⚠️  This will DELETE {len(problematic_restaurants)} restaurants. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            print("\n🗑️  Deleting problematic restaurants...")
            ids_to_delete = [r['id'] for r in problematic_restaurants]
            
            # Delete in batches of 100
            batch_size = 100
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i+batch_size]
                index.delete(ids=batch, namespace="restaurants")
                print(f"   Deleted batch {i//batch_size + 1}/{(len(ids_to_delete)-1)//batch_size + 1}")
            
            print(f"✅ Deleted {len(ids_to_delete)} problematic restaurants")
            print(f"✅ Kept {len(good_restaurants)} good restaurants")
            print("\n💡 Next steps:")
            print("   1. Fix your scraper to extract real dish names")
            print("   2. Re-scrape the problematic restaurants")
            print("   3. Upload using: python backend/restaurant_to_pinecone.py your_data.json")
        else:
            print("❌ Cancelled")
    
    elif choice == "2":
        confirm = input(f"\n⚠️  This will DELETE ALL {len(restaurants)} restaurants. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            print("\n🗑️  Deleting all restaurants...")
            index.delete(delete_all=True, namespace="restaurants")
            print("✅ All restaurants deleted")
            print("\n💡 Next steps:")
            print("   1. Fix your scraper to extract real dish names")
            print("   2. Re-scrape all restaurants")
            print("   3. Upload using: python backend/restaurant_to_pinecone.py your_data.json")
        else:
            print("❌ Cancelled")
    
    elif choice == "3":
        output_file = "problematic_restaurants.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(problematic_restaurants, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Exported to {output_file}")
        print("   Review this file to see which restaurants need re-scraping")
    
    elif choice == "4":
        print("\n👋 Exiting without changes")
    else:
        print("\n❌ Invalid choice")

print("\n" + "=" * 70)
print("🎯 NEXT STEPS FOR FIXING YOUR SCRAPER")
print("=" * 70)
print("""
Your web scraper should extract:

✅ CORRECT:
{
  "name": "Roma Antica",
  "cuisine_types": ["Italian", "Pizza", "Pasta Shops"],  ← Categories
  "menu_items": [                                         ← Actual dishes
    "Margherita Pizza",
    "Pepperoni Pizza", 
    "Fettuccine Alfredo",
    "Caesar Salad",
    "Tiramisu"
  ]
}

❌ WRONG:
{
  "name": "Roma Antica",
  "cuisine_types": ["Italian", "Pizza", "Pasta Shops"],
  "menu_items": ["Italian", "Pizza", "Pasta Shops"]  ← This is wrong!
}

Make sure your scraper:
1. Extracts cuisine categories → cuisine_types
2. Extracts actual dish names from menu → menu_items
3. Does NOT duplicate cuisine_types into menu_items
""")

print("\n✨ Done!")
