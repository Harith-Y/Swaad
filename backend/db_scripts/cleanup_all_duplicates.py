"""
Find and remove duplicates across ALL locations
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index


def main():
    """Find and remove duplicate restaurants across all locations"""
    
    print("Fetching all restaurants from Pinecone...\n")
    
    pc_index = get_pinecone_index()
    
    # Query for ALL restaurants
    dummy_vector = [0.0] * 384
    results = pc_index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        namespace="restaurants"
    )
    
    print(f"Total restaurants fetched: {len(results.get('matches', []))}\n")
    
    # Group by name and city
    restaurants_by_key = defaultdict(list)
    city_stats = defaultdict(int)
    
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        city = meta.get("city", "Unknown")
        name = meta.get("name", "Unknown")
        
        # Create unique key: name + city
        key = f"{name.lower().strip()}|{city.lower().strip()}"
        
        restaurants_by_key[key].append({
            "id": match.get("id"),
            "name": name,
            "city": city,
            "rating": meta.get("avg_rating", 0),
            "dishes": len(meta.get("menu_items", [])),
        })
        
        city_stats[city] += 1
    
    print("Restaurants by city:")
    for city, count in sorted(city_stats.items()):
        print(f"  {city}: {count}")
    print()
    
    # Find duplicates
    duplicates = {key: entries for key, entries in restaurants_by_key.items() if len(entries) > 1}
    
    print(f"Unique restaurant-city combinations: {len(restaurants_by_key)}")
    print(f"Duplicates found: {len(duplicates)}\n")
    
    if duplicates:
        print("="*60)
        print("DUPLICATE RESTAURANTS:")
        print("="*60)
        
        to_delete = []
        to_keep = []
        
        for key, entries in sorted(duplicates.items()):
            # Sort by: dishes count (desc), rating (desc)
            sorted_entries = sorted(entries, key=lambda x: (x['dishes'], x['rating']), reverse=True)
            
            best = sorted_entries[0]
            rest = sorted_entries[1:]
            
            to_keep.append(best)
            to_delete.extend(rest)
            
            print(f"\n{best['name']} - {best['city']} ({len(entries)} copies)")
            print(f"  ✅ Keeping: {best['id']} (rating: {best['rating']}, dishes: {best['dishes']})")
            for dup in rest:
                print(f"  ❌ Deleting: {dup['id']} (rating: {dup['rating']}, dishes: {dup['dishes']})")
        
        print(f"\n{'='*60}")
        print(f"Total to delete: {len(to_delete)}")
        print(f"{'='*60}\n")
        
        if to_delete:
            confirm = input(f"Delete {len(to_delete)} duplicate entries? (y/n): ").strip().lower()
            if confirm == 'y':
                print("\nDeleting duplicates...")
                
                # Delete in batches
                batch_size = 100
                delete_ids = [entry['id'] for entry in to_delete]
                
                for i in range(0, len(delete_ids), batch_size):
                    batch = delete_ids[i:i + batch_size]
                    pc_index.delete(ids=batch, namespace="restaurants")
                    print(f"  Deleted batch {i // batch_size + 1}/{(len(delete_ids) + batch_size - 1) // batch_size}")
                
                print(f"\n✅ Successfully deleted {len(to_delete)} duplicate restaurants!")
                print(f"   Total unique restaurants remaining: {len(restaurants_by_key) - len(duplicates) + len(to_keep)}")
            else:
                print("Aborted.")
        else:
            print("No duplicates to delete!")
    else:
        print("✅ No duplicates found!")


if __name__ == "__main__":
    main()
