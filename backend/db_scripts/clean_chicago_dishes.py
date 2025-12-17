"""
Clean existing dish names in Pinecone database
This fixes formatting artifacts without re-scraping
"""
import os
import sys
import json
import re

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
    """Clean Chicago restaurant dishes in Pinecone"""
    
    print("Fetching Chicago restaurants from Pinecone...\n")
    
    pc_index = get_pinecone_index()
    scraper = MenuURLScraper()
    
    # Query for all Chicago restaurants
    dummy_vector = [0.0] * 384
    results = pc_index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        namespace="restaurants"
    )
    
    chicago_restaurants = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        city = meta.get("city", "")
        if isinstance(city, str) and "chicago" in city.lower():
            menu_items = meta.get("menu_items", [])
            if menu_items:
                chicago_restaurants.append({
                    "id": match.get("id"),
                    "name": meta.get("name"),
                    "menu_items": menu_items,
                    "metadata": meta
                })
    
    print(f"Found {len(chicago_restaurants)} Chicago restaurants with menus\n")
    
    # Clean dishes
    updates = []
    cleaned_count = 0
    
    for i, rest in enumerate(chicago_restaurants):
        name = rest['name']
        original_dishes = rest['menu_items']
        
        # Clean dishes
        cleaned_dishes = clean_dish_list(original_dishes, scraper)
        
        if cleaned_dishes != original_dishes:
            print(f"[{i+1}] {name}")
            print(f"   Original: {len(original_dishes)} dishes")
            print(f"   Cleaned:  {len(cleaned_dishes)} dishes")
            
            # Show sample changes
            if len(original_dishes) > 0:
                print(f"   Sample:")
                for orig, clean in zip(original_dishes[:3], cleaned_dishes[:3]):
                    if orig != clean:
                        print(f"     '{orig}' → '{clean}'")
            print()
            
            # Prepare update
            rest['metadata']['menu_items'] = cleaned_dishes
            updates.append({
                "id": rest['id'],
                "metadata": rest['metadata']
            })
            cleaned_count += 1
    
    print(f"\n{'='*60}")
    print(f"Found {cleaned_count} restaurants needing cleanup")
    print(f"{'='*60}\n")
    
    if updates:
        confirm = input(f"Update {len(updates)} restaurants in Pinecone? (y/n): ").strip().lower()
        if confirm == 'y':
            print("\nUpdating Pinecone...")
            
            # Update in batches
            batch_size = 100
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i + batch_size]
                
                # Pinecone upsert with same ID updates metadata
                upsert_data = []
                for update in batch:
                    # Need to get the vector too (query again)
                    result = pc_index.fetch(ids=[update['id']], namespace="restaurants")
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
            
            print(f"\n✅ Successfully cleaned {len(updates)} Chicago restaurants!")
        else:
            print("Aborted.")
    else:
        print("No updates needed - all dishes are clean!")


if __name__ == "__main__":
    main()
