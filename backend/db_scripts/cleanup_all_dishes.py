"""
Clean dish names for ALL restaurants in Pinecone
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index
from menu_url_scraper import MenuURLScraper


def clean_dish_list(dishes, scraper):
    """Clean a list of dish names"""
    cleaned = []
    for dish in dishes:
        clean_dish = scraper._clean_dish_name(dish)
        if scraper._is_valid_dish_name(clean_dish):
            cleaned.append(clean_dish)
    return cleaned


def main():
    """Clean all restaurant dishes in Pinecone"""
    
    print("Fetching ALL restaurants from Pinecone...\n")
    
    pc_index = get_pinecone_index()
    scraper = MenuURLScraper()
    
    # Query for all restaurants
    dummy_vector = [0.0] * 384
    results = pc_index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        namespace="restaurants"
    )
    
    print(f"Total restaurants fetched: {len(results.get('matches', []))}\n")
    
    # Collect all restaurants with menus
    all_restaurants = []
    city_stats = {}
    
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        city = meta.get("city", "Unknown")
        menu_items = meta.get("menu_items", [])
        
        if menu_items:
            all_restaurants.append({
                "id": match.get("id"),
                "name": meta.get("name"),
                "city": city,
                "menu_items": menu_items,
                "metadata": meta
            })
            
            city_stats[city] = city_stats.get(city, 0) + 1
    
    print("Restaurants with menus by city:")
    for city, count in sorted(city_stats.items()):
        print(f"  {city}: {count}")
    print(f"\nTotal: {len(all_restaurants)} restaurants with menus\n")
    
    # Clean dishes
    updates = []
    cleaned_count = 0
    city_cleaned_stats = {}
    
    for i, rest in enumerate(all_restaurants):
        name = rest['name']
        city = rest['city']
        original_dishes = rest['menu_items']
        
        # Clean dishes
        cleaned_dishes = clean_dish_list(original_dishes, scraper)
        
        if cleaned_dishes != original_dishes:
            city_cleaned_stats[city] = city_cleaned_stats.get(city, 0) + 1
            
            if cleaned_count < 10:  # Show first 10 examples
                print(f"[{cleaned_count + 1}] {name} ({city})")
                print(f"   Original: {len(original_dishes)} dishes")
                print(f"   Cleaned:  {len(cleaned_dishes)} dishes")
                
                # Show sample changes
                changes_shown = 0
                for orig, clean in zip(original_dishes, cleaned_dishes):
                    if orig != clean and changes_shown < 2:
                        print(f"     '{orig}' → '{clean}'")
                        changes_shown += 1
                print()
            
            # Prepare update
            rest['metadata']['menu_items'] = cleaned_dishes
            updates.append({
                "id": rest['id'],
                "metadata": rest['metadata']
            })
            cleaned_count += 1
        
        if (i + 1) % 200 == 0:
            print(f"Progress: {i + 1}/{len(all_restaurants)} processed...")
    
    print(f"\n{'='*60}")
    print(f"Found {cleaned_count} restaurants needing cleanup")
    print(f"\nBy city:")
    for city, count in sorted(city_cleaned_stats.items()):
        print(f"  {city}: {count}")
    print(f"{'='*60}\n")
    
    if updates:
        confirm = input(f"Update {len(updates)} restaurants in Pinecone? (y/n): ").strip().lower()
        if confirm == 'y':
            print("\nUpdating Pinecone...")
            
            # Update in batches
            batch_size = 100
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i + batch_size]
                
                # Fetch vectors for each update
                upsert_data = []
                batch_ids = [update['id'] for update in batch]
                result = pc_index.fetch(ids=batch_ids, namespace="restaurants")
                
                for update in batch:
                    if result.get('vectors') and update['id'] in result['vectors']:
                        vector_data = result['vectors'][update['id']]
                        upsert_data.append({
                            "id": update['id'],
                            "values": vector_data.get('values'),
                            "metadata": update['metadata']
                        })
                
                if upsert_data:
                    pc_index.upsert(vectors=upsert_data, namespace="restaurants")
                    print(f"  Updated batch {i // batch_size + 1}/{(len(updates) + batch_size - 1) // batch_size}")
            
            print(f"\n✅ Successfully cleaned {len(updates)} restaurants across all locations!")
        else:
            print("Aborted.")
    else:
        print("✅ No updates needed - all dishes are clean!")


if __name__ == "__main__":
    main()
