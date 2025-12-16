"""
Comprehensive Restaurant & Dish Population Script

This script properly populates restaurant dishes into Pinecone for multiple locations.
It uses:
1. Yelp API for restaurant data
2. Menu URL scraping for detailed dish lists
3. Proper error handling and retry logic
4. Batch processing for efficiency

Usage:
    python populate_dishes_comprehensive.py --locations "San Francisco, CA" "San Jose, CA" "Oakland, CA" "New York, NY" --limit 100
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yelp_api_client import YelpAPIClient
from menu_url_scraper import MenuURLScraper
from restaurant_to_pinecone import process_restaurants, upload_to_pinecone


def populate_location(location: str, limit: int = 100, category: str = "restaurants") -> Dict[str, Any]:
    """
    Populate restaurants with dishes for a specific location
    
    Args:
        location: Location string (e.g., "San Francisco, CA")
        limit: Number of restaurants to fetch
        category: Category to search (default: "restaurants")
    
    Returns:
        Dictionary with stats about the operation
    """
    print("\n" + "="*80)
    print(f"PROCESSING: {location}")
    print("="*80)
    
    stats = {
        "location": location,
        "total_found": 0,
        "with_menu_url": 0,
        "successfully_scraped": 0,
        "uploaded_to_pinecone": 0,
        "failed": 0
    }
    
    try:
        # Step 1: Get restaurants from Yelp
        print(f"\n📡 Fetching restaurants from Yelp API...")
        yelp_client = YelpAPIClient()
        restaurants = yelp_client.search_and_get_menu_urls(category, location, limit)
        
        if not restaurants:
            print(f"❌ No restaurants found for {location}")
            return stats
        
        stats["total_found"] = len(restaurants)
        stats["with_menu_url"] = sum(1 for r in restaurants if r.get("menu_url"))
        
        print(f"✅ Found {len(restaurants)} restaurants")
        print(f"   {stats['with_menu_url']} have menu URLs")
        
        # Step 2: Scrape menus
        print(f"\n🍽️  Scraping menus...")
        menu_scraper = MenuURLScraper()
        processed_restaurants = []
        
        for i, restaurant in enumerate(restaurants, 1):
            name = restaurant.get("name")
            menu_url = restaurant.get("menu_url")
            
            print(f"\n[{i}/{len(restaurants)}] {name}")
            
            if menu_url:
                try:
                    dishes = menu_scraper.scrape_menu_url(menu_url)
                    if dishes:
                        restaurant["menu_items"] = dishes
                        restaurant["popular_dishes"] = dishes[:10]  # First 10 as popular
                        stats["successfully_scraped"] += 1
                        print(f"   ✅ Found {len(dishes)} dishes")
                    else:
                        print(f"   ⚠️  No dishes extracted")
                        restaurant["menu_items"] = []
                        restaurant["popular_dishes"] = []
                        stats["failed"] += 1
                except Exception as e:
                    print(f"   ❌ Error scraping: {e}")
                    restaurant["menu_items"] = []
                    restaurant["popular_dishes"] = []
                    stats["failed"] += 1
            else:
                print(f"   ⚠️  No menu URL")
                restaurant["menu_items"] = []
                restaurant["popular_dishes"] = []
            
            processed_restaurants.append(restaurant)
            
            # Rate limiting - small delay between requests
            time.sleep(0.5)
        
        # Step 3: Save intermediate results
        location_safe = location.replace(", ", "_").replace(" ", "_")
        output_file = f"scraped_{location_safe}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_restaurants, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved to {output_file}")
        
        # Step 4: Upload to Pinecone
        restaurants_with_menus = [r for r in processed_restaurants if r.get("menu_items")]
        
        if restaurants_with_menus:
            print(f"\n📤 Uploading {len(restaurants_with_menus)} restaurants to Pinecone...")
            try:
                vectors = process_restaurants(restaurants_with_menus)
                upload_to_pinecone(vectors)
                stats["uploaded_to_pinecone"] = len(vectors)
                print(f"✅ Successfully uploaded {len(vectors)} restaurants")
            except Exception as e:
                print(f"❌ Error uploading to Pinecone: {e}")
        else:
            print("\n⚠️  No restaurants with menus to upload")
        
    except Exception as e:
        print(f"\n❌ Error processing {location}: {e}")
        import traceback
        traceback.print_exc()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Populate restaurant dishes for multiple locations"
    )
    
    parser.add_argument(
        "--locations",
        type=str,
        nargs="+",
        default=["San Francisco, CA", "San Jose, CA", "Oakland, CA", "New York, NY"],
        help="List of locations to populate"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of restaurants per location (default: 100)"
    )
    
    parser.add_argument(
        "--category",
        type=str,
        default="restaurants",
        help="Category to search (default: restaurants)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RESTAURANT DISH POPULATION")
    print("="*80)
    print(f"\nLocations: {', '.join(args.locations)}")
    print(f"Limit per location: {args.limit}")
    print(f"Category: {args.category}")
    
    all_stats = []
    
    # Process each location
    for location in args.locations:
        stats = populate_location(location, args.limit, args.category)
        all_stats.append(stats)
        
        # Brief pause between locations
        time.sleep(2)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    total_found = sum(s["total_found"] for s in all_stats)
    total_with_menu = sum(s["with_menu_url"] for s in all_stats)
    total_scraped = sum(s["successfully_scraped"] for s in all_stats)
    total_uploaded = sum(s["uploaded_to_pinecone"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total restaurants found: {total_found}")
    print(f"   With menu URLs: {total_with_menu}")
    print(f"   Successfully scraped: {total_scraped}")
    print(f"   Uploaded to Pinecone: {total_uploaded}")
    print(f"   Failed: {total_failed}")
    
    print(f"\n📍 By Location:")
    for stats in all_stats:
        print(f"\n   {stats['location']}:")
        print(f"      Found: {stats['total_found']}")
        print(f"      Scraped: {stats['successfully_scraped']}")
        print(f"      Uploaded: {stats['uploaded_to_pinecone']}")
    
    print("\n✅ Population complete!")


if __name__ == "__main__":
    main()
