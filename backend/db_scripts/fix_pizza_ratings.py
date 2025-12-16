"""
Fix missing ratings specifically for pizza restaurants

This script:
1. Uses semantic search to find pizza restaurants
2. Fetches their ratings from Yelp API  
3. Updates Pinecone with correct ratings
"""

import os
import sys
import json
import time
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yelp_api_client import YelpAPIClient

# Load environment
load_dotenv()


def fix_pizza_ratings():
    """Fix missing ratings for pizza restaurants"""
    
    # Initialize clients
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "menu-buddy")
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    yelp_client = YelpAPIClient()
    
    print("🍕 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pizza_vec = model.encode("pizza").tolist()
    
    print("🔍 Fetching pizza restaurants from Pinecone...")
    
    # Query for pizza restaurants
    results = index.query(
        vector=pizza_vec,
        top_k=100,
        include_metadata=True,
        namespace="restaurants"
    )
    
    print(f"✅ Retrieved {len(results.matches)} pizza-related restaurants\n")
    
    # Find restaurants with missing ratings
    missing_ratings = []
    has_ratings = 0
    
    for match in results.matches:
        meta = match.metadata
        rating = meta.get("avg_rating", 0.0)
        name = meta.get("name", "Unknown")
        
        if rating == 0.0:
            # Extract Yelp ID from vector id (format: "restaurant:YELP_ID")
            vector_id = match.id
            yelp_id = None
            
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
                "metadata": meta,
                "values": match.values  # Keep original embedding
            })
        else:
            has_ratings += 1
    
    print(f"✅ {has_ratings} pizza restaurants already have ratings")
    print(f"⚠️  {len(missing_ratings)} pizza restaurants need rating updates\n")
    
    if not missing_ratings:
        print("✅ All pizza restaurants have ratings!")
        return
    
    print(f"🔧 Fetching ratings from Yelp API...")
    print("=" * 80)
    
    updated_count = 0
    failed_count = 0
    updates_to_apply = []
    
    for i, rest in enumerate(missing_ratings, 1):
        name = rest["name"]
        city = rest["city"]
        yelp_id = rest["yelp_id"]
        
        print(f"\n[{i}/{len(missing_ratings)}] {name} ({city})")
        
        try:
            # Get business details from Yelp
            details = yelp_client.get_business_details(yelp_id)
            
            if details and "rating" in details:
                rating = details.get("rating", 0.0)
                print(f"   ✅ Found rating: {rating}/5")
                
                # Update metadata
                updated_meta = rest["metadata"].copy()
                updated_meta["avg_rating"] = float(rating)
                
                updates_to_apply.append({
                    "id": rest["vector_id"],
                    "metadata": updated_meta
                })
                
                updated_count += 1
            else:
                print(f"   ⚠️  No rating found in Yelp API")
                failed_count += 1
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed_count += 1
    
    # Apply updates
    if updates_to_apply:
        print(f"\n📤 Updating {len(updates_to_apply)} restaurants in Pinecone...")
        
        success_count = 0
        fail_count = 0
        
        # Update each restaurant individually (metadata only)
        for update in updates_to_apply:
            try:
                index.update(
                    id=update["id"],
                    set_metadata=update["metadata"],
                    namespace="restaurants"
                )
                success_count += 1
                if success_count % 10 == 0:
                    print(f"   ✅ Updated {success_count}/{len(updates_to_apply)} restaurants...")
            except Exception as e:
                print(f"   ❌ Failed to update {update['id']}: {e}")
                fail_count += 1
        
        print(f"   ✅ Successfully updated {success_count}/{len(updates_to_apply)} restaurants!")
    
    # Final report
    print(f"\n{'='*80}")
    print(f"📊 FINAL REPORT")
    print(f"{'='*80}")
    print(f"   Pizza restaurants checked: {len(results.matches)}")
    print(f"   With missing ratings: {len(missing_ratings)}")
    print(f"   Successfully updated: {updated_count}")
    print(f"   Failed to update: {failed_count}")
    print(f"\n✅ Pizza rating fix complete!")


if __name__ == "__main__":
    print("="*80)
    print("🍕 PIZZA RESTAURANT RATING FIXER")
    print("="*80)
    print("\nThis script will:")
    print("1. Find pizza restaurants in Pinecone with 0.0 rating")
    print("2. Fetch their ratings from Yelp API")
    print("3. Update Pinecone with correct ratings")
    
    input("\nPress ENTER to continue or CTRL+C to cancel...")
    
    fix_pizza_ratings()
