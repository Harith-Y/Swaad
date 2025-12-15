"""
Clean menu data in Pinecone by filtering out:
1. Menu category headers (Appetizers, HANDMADE PASTA, etc.)
2. Website elements (Email Signup, Reservations, etc.)
3. Generic terms that aren't actual dishes
4. Keep only real dish names

Usage:
    python clean_menu_data.py --dry-run  # Preview changes
    python clean_menu_data.py --apply    # Apply changes
"""
import os
import sys
import argparse
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "menu-buddy")

# Categories and non-dish items to filter out
NON_DISH_KEYWORDS = {
    # Menu categories
    'appetizer', 'appetizers', 'antipasti', 'starters',
    'mains', 'entrees', 'secondi', 'main course',
    'desserts', 'dolci', 'sweets',
    'salads', 'insalata',
    'soups', 'zuppa',
    'pasta', 'handmade pasta', 'fresh pasta',
    'pizza', 'pizzas',
    'toppings', 'add-ons', 'extras',
    'sides', 'side dishes',
    'fritti', 'fried',
    
    # Drinks categories
    'beverages', 'drinks', 'cocktails', 'beer', 'wine',
    'draft beer', 'bottled beer', 'canned beer',
    'bottled and canned beers',
    'bubbles', 'bianco', 'rosso',
    'vino', 'vino della casa',
    'spirit free', 'non-alcoholic',
    'caffe', 'coffee', 'tea',
    
    # Website/menu elements
    'email signup', 'reservations', 'order online',
    'call to order', 'menu', 'specials',
    'daily specials', 'chef specials',
    
    # Generic terms
    'classic', 'classics', 'traditional',
    'seasonal', 'market fresh',
    'gluten free options', 'vegan options',
    
    # Generic gluten-free/dietary categories (not specific dishes)
    'gluten free pizza', 'gluten free insalata', 'gluten free desserts',
    'gluten free salads', 'gluten free pasta',
    
    # Rubirosa-specific generic terms
    'piatto di antipasti', 'rubirosa classics',
    'dessert cocktails',
}

def is_likely_non_dish(item_name):
    """Check if an item is likely a category or non-dish element."""
    item_lower = item_name.lower().strip()
    
    # Exact matches
    if item_lower in NON_DISH_KEYWORDS:
        return True
    
    # Very short names (likely categories)
    if len(item_lower) <= 4 and item_lower.isalpha():
        # Exception: allow 4-letter food words
        if item_lower not in {'beef', 'pork', 'duck', 'rice', 'tuna', 'brie'}:
            return True
    
    # All caps with spaces (menu headers)
    if item_name.isupper() and ' ' in item_name:
        return True
    
    # Ends with category markers
    if item_lower.endswith((' menu', ' selection', ' items', ' dishes')):
        return True
    
    # Generic gluten-free/vegan without specific dish name
    dietary_prefixes = ['gluten free ', 'vegan ', 'vegetarian ']
    generic_foods = ['pizza', 'pasta', 'salad', 'insalata', 'desserts', 'appetizers', 'entrees']
    for prefix in dietary_prefixes:
        if item_lower.startswith(prefix):
            remainder = item_lower[len(prefix):]
            if remainder in generic_foods:
                return True
    
    return False


def clean_menu_items(menu_items):
    """Filter out non-dish items from menu."""
    if not menu_items:
        return []
    
    cleaned = []
    removed = []
    
    for item in menu_items:
        if not isinstance(item, str):
            continue
        
        item = item.strip()
        if not item:
            continue
        
        if is_likely_non_dish(item):
            removed.append(item)
        else:
            cleaned.append(item)
    
    return cleaned, removed


def main():
    parser = argparse.ArgumentParser(description='Clean menu data in Pinecone')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--apply', action='store_true', help='Apply changes to Pinecone')
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        print("Please specify either --dry-run or --apply")
        sys.exit(1)
    
    mode = "DRY RUN (Preview Only)" if args.dry_run else "APPLYING CHANGES"
    
    print("=" * 80)
    print(f"🧹 CLEANING MENU DATA - {mode}")
    print("=" * 80)
    
    # Connect to Pinecone
    print("\n🔌 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    print(f"✅ Connected to index: {PINECONE_INDEX}")
    
    # Fetch all restaurants
    print("\n📥 Fetching all restaurants...")
    query_vec = [0.1] * 384
    result = index.query(
        vector=query_vec,
        top_k=10000,
        include_metadata=True,
        namespace="restaurants"
    )
    
    restaurants = result.get('matches', [])
    print(f"✅ Found {len(restaurants)} restaurants\n")
    
    # Analyze and clean
    print("=" * 80)
    print("📊 ANALYSIS")
    print("=" * 80)
    
    updates = []
    total_removed = 0
    restaurants_affected = 0
    
    for restaurant in restaurants:
        meta = restaurant.get('metadata', {})
        restaurant_id = restaurant.get('id')
        restaurant_name = meta.get('name', 'Unknown')
        menu_items = meta.get('menu_items', [])
        
        if not menu_items:
            continue
        
        # Clean menu items
        cleaned, removed = clean_menu_items(menu_items)
        
        if removed:
            restaurants_affected += 1
            total_removed += len(removed)
            
            print(f"\n🏪 {restaurant_name}")
            print(f"   Original: {len(menu_items)} items")
            print(f"   Cleaned: {len(cleaned)} items")
            print(f"   ❌ Removed {len(removed)} items:")
            for item in removed:
                print(f"      - {item}")
            
            if len(cleaned) < 3:
                print(f"   ⚠️  WARNING: Only {len(cleaned)} dishes left - may need manual review")
            
            # Prepare update
            if args.apply:
                # Update metadata
                updated_meta = meta.copy()
                updated_meta['menu_items'] = cleaned
                
                # Also update popular_dishes if needed
                if 'popular_dishes' in updated_meta:
                    popular = updated_meta['popular_dishes']
                    if isinstance(popular, list):
                        cleaned_popular, _ = clean_menu_items(popular)
                        updated_meta['popular_dishes'] = cleaned_popular[:3]
                
                updates.append({
                    'id': restaurant_id,
                    'metadata': updated_meta
                })
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"Total restaurants: {len(restaurants)}")
    print(f"Restaurants affected: {restaurants_affected}")
    print(f"Total items removed: {total_removed}")
    
    # Apply updates
    if args.apply and updates:
        print(f"\n✍️  Applying {len(updates)} metadata updates to Pinecone...")
        
        # Update metadata only (no need for vector data)
        for i, update in enumerate(updates, 1):
            try:
                index.update(
                    id=update['id'],
                    set_metadata=update['metadata'],
                    namespace="restaurants"
                )
                print(f"   Updated {i}/{len(updates)} restaurants")
            except Exception as e:
                print(f"   ❌ Failed to update {update['id']}: {str(e)}")
        
        print(f"✅ All updates applied!")
    elif args.dry_run:
        print(f"\n💡 This was a dry run. Use --apply to make changes.")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
