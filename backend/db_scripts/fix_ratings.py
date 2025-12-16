"""
Fix missing ratings for existing restaurants in Pinecone

This script:
1. Queries all restaurants from Pinecone  
2. For those with 0.0 rating, fetches rating from Yelp API
3. Updates Pinecone with correct ratings
"""

import os
import sys
import json
import time
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yelp_api_client import YelpAPIClient

# Load environment
load_dotenv()


def fix_missing_ratings(batch_size: int = 50):
    """Fix missing ratings in Pinecone"""
    
    # Initialize clients
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "menu-buddy")
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    yelp_client = YelpAPIClient()
    
    print("🔍 Fetching all restaurants from Pinecone...")
    
    # Query all restaurants in batches
    all_restaurants = []
    dummy_vector = [0.0] * 384
    
    # Fetch in multiple queries to get more results
    for i in range(15):  # 15 queries x 100 = 1500 potential restaurants
        results = index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True,
            namespace="restaurants",
            skip=i * 100
        )
        
        if not results.matches:
            break
        
        all_restaurants.extend(results.matches)
        print(f"   Fetched {len(all_restaurants)} restaurants so far...")
        time.sleep(0.1)
    
    print(f"\n✅ Total restaurants retrieved: {len(all_restaurants)}")
    
    # Find restaurants with missing ratings
    missing_ratings = []
    for match in all_restaurants:
        meta = match.metadata
        rating = meta.get("avg_rating", 0.0)
        
        if rating == 0.0:
            name = meta.get("name", "Unknown")
            
            # Try to get ID from vector id (format: "restaurant:YELP_ID")
            vector_id = match.id
            yelp_id = None
            
            # The vector ID should be in format "restaurant:123" or just the yelp ID
            if ":" in vector_id:
                parts = vector_id.split(":", 1)
                if len(parts) == 2:
                    yelp_id = parts[1]
            else:
                yelp_id = vector_id
            
            try:
                location_json = meta.get("location_json", "{}")
                if location_json:
                    location = json.loads(location_json)
                    city = location.get("city", "Unknown")
                else:
                    city = "Unknown"
            except:
                city = "Unknown"
            
            missing_ratings.append({
                "vector_id": vector_id,
                "yelp_id": yelp_id,
                "name": name,
                "city": city,
                "metadata": meta
            })
    
    print(f"\n⚠️  Found {len(missing_ratings)} restaurants with 0.0 rating")
    
    if not missing_ratings:
        print("✅ All restaurants have ratings!")
        return
    
    print(f"\n🔧 Fetching ratings from Yelp API...")
    print("=" * 80)
    
    updated_count = 0
    failed_count = 0
    updates_to_apply = []
    
    for i, rest in enumerate(missing_ratings, 1):
        name = rest["name"]
        yelp_id = rest["yelp_id"]
        
        print(f"\n[{i}/{len(missing_ratings)}] {name}")
        
        try:
            # Get business details from Yelp
            details = yelp_client.get_business_details(yelp_id)
            
            if details and "rating" in details:
                rating = details.get("rating", 0.0)
                print(f"   ✅ Found rating: {rating}/5")
                
                # Prepare metadata update
                updated_meta = rest["metadata"].copy()
                updated_meta["avg_rating"] = float(rating)
                
                updates_to_apply.append({
                    "id": rest["vector_id"],
                    "set_metadata": updated_meta
                })
                
                updated_count += 1
            else:
                print(f"   ⚠️  No rating found in Yelp")
                failed_count += 1
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed_count += 1
    
    # Apply updates in batches
    if updates_to_apply:
        print(f"\n📤 Updating {len(updates_to_apply)} restaurants in Pinecone...")
        
        for i in range(0, len(updates_to_apply), batch_size):
            batch = updates_to_apply[i:i+batch_size]
            
            try:
                index.update(
                    vectors=[{
                        "id": update["id"],
                        "set_metadata": update["set_metadata"]
                    } for update in batch],
                    namespace="restaurants"
                )
                print(f"   ✅ Updated batch {i//batch_size + 1}/{(len(updates_to_apply)-1)//batch_size + 1}")
                time.sleep(0.5)
            except Exception as e:
                print(f"   ❌ Batch update failed: {e}")
                # Try individual updates for this batch
                for update in batch:
                    try:
                        index.update(
                            id=update["id"],
                            set_metadata=update["set_metadata"],
                            namespace="restaurants"
                        )
                    except Exception as e2:
                        print(f"      ❌ Failed to update {update['id']}: {e2}")
    
    # Final report
    print(f"\n{'='*80}")
    print(f"📊 FINAL REPORT")
    print(f"{'='*80}")
    print(f"   Total restaurants checked: {len(all_restaurants)}")
    print(f"   With missing ratings: {len(missing_ratings)}")
    print(f"   Successfully updated: {updated_count}")
    print(f"   Failed to update: {failed_count}")
    print(f"\n✅ Rating fix complete!")


if __name__ == "__main__":
    print("="*80)
    print("🔧 RESTAURANT RATING FIXER")
    print("="*80)
    print("\nThis script will:")
    print("1. Find all restaurants in Pinecone with 0.0 rating")
    print("2. Fetch their ratings from Yelp API")
    print("3. Update Pinecone with correct ratings")
    print("\nThis may take a while depending on how many restaurants need updating...")
    
    input("\n Press ENTER to continue or CTRL+C to cancel...")
    
    fix_missing_ratings()
