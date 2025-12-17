"""
Re-scrape Chicago restaurants with fixed menu scraper
This will update existing restaurants with clean dish names
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index
from menu_url_scraper import MenuURLScraper
from restaurant_to_pinecone import process_restaurants, upload_to_pinecone


def main():
    """Re-scrape Chicago restaurants"""
    
    # Load saved restaurants
    json_file = "master_restaurants_20251217_033916.json"
    
    print("Loading Chicago restaurants...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for Chicago restaurants with menu URLs
    chicago_with_menus = []
    for r in data:
        location = r.get('location', {})
        if isinstance(location, dict):
            city = location.get('city', '')
            menu_url = r.get('menu_url')
            if isinstance(city, str) and 'chicago' in city.lower() and menu_url:
                chicago_with_menus.append(r)
    
    print(f"Found {len(chicago_with_menus)} Chicago restaurants with menu URLs\n")
    
    # Initialize scraper
    scraper = MenuURLScraper()
    
    # Re-scrape dishes
    updated = 0
    skipped = 0
    
    for i, restaurant in enumerate(chicago_with_menus):
        name = restaurant.get('name', 'Unknown')
        menu_url = restaurant.get('menu_url')
        
        print(f"\n[{i+1}/{len(chicago_with_menus)}] {name}")
        print(f"   URL: {menu_url}")
        
        try:
            # Re-scrape menu
            dishes = scraper.scrape_menu_url(menu_url)
            
            if dishes:
                print(f"   ✅ Found {len(dishes)} clean dishes")
                restaurant['menu_items'] = dishes
                updated += 1
            else:
                print(f"   ⚠️  No dishes extracted")
                skipped += 1
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            skipped += 1
        
        if (i + 1) % 20 == 0:
            print(f"\n--- Progress: {i + 1}/{len(chicago_with_menus)} ({updated} updated, {skipped} skipped) ---\n")
    
    print(f"\n{'='*60}")
    print(f"Re-scraping complete!")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")
    print(f"{'='*60}\n")
    
    if updated > 0:
        # Re-upload updated restaurants
        print("Re-uploading to Pinecone...")
        chicago_updated = [r for r in data if 'menu_items' in r and r in chicago_with_menus]
        
        vectors = process_restaurants(chicago_updated[:updated])  # Only updated ones
        upload_to_pinecone(vectors)
        
        print(f"\n✅ Successfully re-uploaded {updated} Chicago restaurants with clean dishes!")


if __name__ == "__main__":
    main()
