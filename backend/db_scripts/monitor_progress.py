"""
Progress Monitor for Restaurant Population

This script monitors the progress of ongoing restaurant population
without interrupting the main process.

Usage:
    # In a separate terminal
    python monitor_progress.py
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()


def get_pinecone_stats():
    """Get current Pinecone statistics"""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("menu-buddy")
        stats = index.describe_index_stats()
        
        restaurant_count = stats.namespaces.get("restaurants", {}).get("vector_count", 0)
        ingredient_count = stats.namespaces.get("ingredients", {}).get("vector_count", 0)
        total = stats.total_vector_count
        
        return {
            "restaurants": restaurant_count,
            "ingredients": ingredient_count,
            "total": total
        }
    except Exception as e:
        return {"error": str(e)}


def check_json_files():
    """Check for JSON output files"""
    files = []
    
    patterns = [
        "master_restaurants_*.json",
        "scraped_*.json",
        "enhanced_*.json",
        "pizza_restaurants.json"
    ]
    
    import glob
    for pattern in patterns:
        matching = glob.glob(pattern)
        files.extend(matching)
    
    return files


def get_file_info(filename):
    """Get info about a JSON file"""
    try:
        import json
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            count = len(data)
            with_menus = sum(1 for r in data if r.get("menu_items"))
            total_dishes = sum(len(r.get("menu_items", [])) for r in data)
            
            return {
                "restaurants": count,
                "with_menus": with_menus,
                "total_dishes": total_dishes
            }
    except:
        return None


def monitor():
    """Main monitoring function"""
    print("\n" + "="*80)
    print("📊 RESTAURANT POPULATION PROGRESS MONITOR")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop monitoring\n")
    
    initial_stats = get_pinecone_stats()
    initial_restaurants = initial_stats.get("restaurants", 0)
    
    print(f"🔢 Initial Count: {initial_restaurants} restaurants in Pinecone\n")
    print("="*80)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Check #{iteration}")
            print("-" * 80)
            
            # Pinecone stats
            stats = get_pinecone_stats()
            if "error" not in stats:
                current = stats["restaurants"]
                added = current - initial_restaurants
                
                print(f"📦 Pinecone Database:")
                print(f"   Restaurants: {current} (+{added} since start)")
                print(f"   Ingredients: {stats['ingredients']}")
                print(f"   Total vectors: {stats['total']}")
            else:
                print(f"❌ Pinecone error: {stats['error']}")
            
            # Check JSON files
            json_files = check_json_files()
            if json_files:
                print(f"\n📄 Output Files Found: {len(json_files)}")
                for filename in json_files:
                    info = get_file_info(filename)
                    if info:
                        print(f"\n   {filename}:")
                        print(f"      Restaurants: {info['restaurants']}")
                        print(f"      With menus: {info['with_menus']}")
                        print(f"      Total dishes: {info['total_dishes']}")
            else:
                print(f"\n📄 No output files yet")
            
            # Progress estimate
            if "error" not in stats and added > 0:
                print(f"\n📈 Progress:")
                print(f"   Added in this run: {added} restaurants")
                if iteration > 1:
                    rate = added / (iteration * 30)  # restaurants per second
                    print(f"   Rate: ~{rate:.2f} restaurants/sec")
            
            print("-" * 80)
            print("Waiting 30 seconds... (Ctrl+C to stop)")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("🛑 Monitoring stopped")
        print("="*80)
        
        final_stats = get_pinecone_stats()
        if "error" not in final_stats:
            final_count = final_stats["restaurants"]
            total_added = final_count - initial_restaurants
            
            print(f"\n📊 Final Statistics:")
            print(f"   Started with: {initial_restaurants} restaurants")
            print(f"   Ended with: {final_count} restaurants")
            print(f"   Total added: {total_added} restaurants")
        
        print("\n✅ You can close this monitor anytime")
        print("   The main population script continues independently\n")


if __name__ == "__main__":
    monitor()
