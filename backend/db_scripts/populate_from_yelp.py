"""
Complete Workflow: Yelp API → Menu Scraping → Pinecone Upload

This script implements the full workflow:
1. Use Yelp AI API to search for restaurants and get menuUrl
2. Scrape menuUrl to extract actual dish names
3. Transform data and upload to Pinecone with taste vectors

Requirements:
    pip install requests beautifulsoup4 pypdf2 pillow sentence-transformers groq python-dotenv

Environment Variables:
    YELP_API_KEY=your_yelp_api_key
    GROQ_API_KEY=your_groq_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX=menu-buddy

Usage:
    # Search and populate from scratch
    python populate_from_yelp.py --search "restaurants" --location "San Francisco" --limit 50
    
    # Process from existing restaurant list with menu URLs
    python populate_from_yelp.py --file restaurants_with_menu_urls.json
    
    # Process specific restaurant IDs
    python populate_from_yelp.py --ids "restaurant-id-1,restaurant-id-2"
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
from pathlib import Path

# Import our modules
from yelp_api_client import YelpAPIClient
from menu_url_scraper import MenuURLScraper
from restaurant_to_pinecone import process_restaurants, upload_to_pinecone


def populate_from_search(term: str, location: str, limit: int = 20) -> List[Dict]:
    """
    Complete workflow: Search → Get menu URLs → Scrape → Upload
    
    Args:
        term: Search term (e.g., "restaurants", "italian food")
        location: Location (e.g., "San Francisco")
        limit: Number of restaurants to process
    
    Returns:
        List of processed restaurant data
    """
    print("\n" + "="*60)
    print("COMPLETE WORKFLOW: YELP API → MENU SCRAPING → PINECONE")
    print("="*60)
    
    # Step 1: Get restaurants with menu URLs from Yelp API
    print("\n📡 STEP 1: Fetching restaurants from Yelp API...")
    yelp_client = YelpAPIClient()
    restaurants = yelp_client.search_and_get_menu_urls(term, location, limit)
    
    if not restaurants:
        print("❌ No restaurants found")
        return []
    
    print(f"✅ Found {len(restaurants)} restaurants")
    
    # Step 2: Scrape menu URLs to get actual dishes
    print("\n🍽️  STEP 2: Scraping menu URLs for dishes...")
    menu_scraper = MenuURLScraper()
    
    processed_restaurants = []
    scraped_count = 0
    
    for i, restaurant in enumerate(restaurants, 1):
        name = restaurant.get("name")
        menu_url = restaurant.get("menu_url")
        
        print(f"\n[{i}/{len(restaurants)}] {name}")
        
        if menu_url:
            dishes = menu_scraper.scrape_menu_url(menu_url)
            if dishes:
                restaurant["menu_items"] = dishes
                restaurant["popular_dishes"] = dishes[:5]  # First 5 as popular
                scraped_count += 1
                print(f"   ✅ Found {len(dishes)} dishes")
            else:
                print(f"   ⚠️  No dishes extracted from menu URL")
                restaurant["menu_items"] = []
                restaurant["popular_dishes"] = []
        else:
            print(f"   ⚠️  No menu URL available")
            restaurant["menu_items"] = []
            restaurant["popular_dishes"] = []
        
        processed_restaurants.append(restaurant)
    
    print(f"\n✅ Successfully scraped menus for {scraped_count}/{len(restaurants)} restaurants")
    
    # Save intermediate results
    output_file = "scraped_restaurants.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_restaurants, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved scraped data to {output_file}")
    
    # Step 3: Transform and upload to Pinecone
    print("\n📤 STEP 3: Uploading to Pinecone with taste vectors...")
    
    # Filter restaurants that have menu items
    restaurants_with_menus = [r for r in processed_restaurants if r.get("menu_items")]
    
    if not restaurants_with_menus:
        print("❌ No restaurants with menu items to upload")
        return processed_restaurants
    
    print(f"   Processing {len(restaurants_with_menus)} restaurants with menus...")
    vectors = process_restaurants(restaurants_with_menus)
    
    print(f"   Uploading {len(vectors)} vectors to Pinecone...")
    upload_to_pinecone(vectors)
    
    print("\n" + "="*60)
    print("✅ WORKFLOW COMPLETE!")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   - Total restaurants found: {len(restaurants)}")
    print(f"   - Restaurants with menu URLs: {sum(1 for r in restaurants if r.get('menu_url'))}")
    print(f"   - Successfully scraped menus: {scraped_count}")
    print(f"   - Uploaded to Pinecone: {len(vectors)}")
    print()
    
    return processed_restaurants


def populate_from_file(file_path: str) -> List[Dict]:
    """
    Process restaurants from a JSON file (that already has menu URLs)
    
    Args:
        file_path: Path to JSON file with restaurant data
    
    Returns:
        List of processed restaurant data
    """
    print(f"\n📂 Loading restaurants from {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        restaurants = json.load(f)
    
    print(f"✅ Loaded {len(restaurants)} restaurants")
    
    # Check if restaurants already have menu_items
    has_menu_items = all(r.get("menu_items") for r in restaurants)
    
    if has_menu_items:
        print("✅ Restaurants already have menu items")
        print("   Skipping scraping, proceeding to Pinecone upload...")
        
        # Upload to Pinecone
        print("\n📤 Uploading to Pinecone...")
        vectors = process_restaurants(restaurants)
        upload_to_pinecone(vectors)
        
        return restaurants
    
    # Otherwise, scrape menu URLs
    print("\n🍽️  Scraping menu URLs...")
    menu_scraper = MenuURLScraper()
    
    for i, restaurant in enumerate(restaurants, 1):
        name = restaurant.get("name")
        menu_url = restaurant.get("menu_url")
        
        print(f"\n[{i}/{len(restaurants)}] {name}")
        
        if menu_url:
            dishes = menu_scraper.scrape_menu_url(menu_url)
            restaurant["menu_items"] = dishes
            restaurant["popular_dishes"] = dishes[:5]
        else:
            restaurant["menu_items"] = []
            restaurant["popular_dishes"] = []
    
    # Save updated file
    output_file = "scraped_restaurants.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(restaurants, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved to {output_file}")
    
    # Upload to Pinecone
    restaurants_with_menus = [r for r in restaurants if r.get("menu_items")]
    if restaurants_with_menus:
        print(f"\n📤 Uploading {len(restaurants_with_menus)} restaurants to Pinecone...")
        vectors = process_restaurants(restaurants_with_menus)
        upload_to_pinecone(vectors)
    
    return restaurants


def main():
    parser = argparse.ArgumentParser(description="Populate Pinecone from Yelp API + Menu Scraping")
    
    parser.add_argument("--search", type=str, help="Search term (e.g., 'restaurants')")
    parser.add_argument("--location", type=str, help="Location (e.g., 'San Francisco')")
    parser.add_argument("--limit", type=int, default=20, help="Number of restaurants to fetch")
    parser.add_argument("--file", type=str, help="JSON file with restaurant data")
    
    args = parser.parse_args()
    
    if args.file:
        # Process from file
        populate_from_file(args.file)
    
    elif args.search and args.location:
        # Search and process
        populate_from_search(args.search, args.location, args.limit)
    
    else:
        print("❌ Error: Provide either --file or both --search and --location")
        print("\nExamples:")
        print("  python populate_from_yelp.py --search 'restaurants' --location 'San Francisco' --limit 50")
        print("  python populate_from_yelp.py --file restaurants_with_menu_urls.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
