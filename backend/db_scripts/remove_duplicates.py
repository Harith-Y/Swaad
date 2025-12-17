"""
Remove duplicate restaurants from Pinecone
Keep the best version (most dishes, highest rating)
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index


def main():
    """Remove duplicate restaurants"""
    
    print("Fetching all Chicago restaurants...\n")
    
    pc_index = get_pinecone_index()
    
    # Query for all restaurants
    dummy_vector = [0.0] * 384
    results = pc_index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        namespace="restaurants"
    )
    
    # Group by name and city
    restaurants_by_name = defaultdict(list)
    
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        city = meta.get("city", "")
        name = meta.get("name", "")
        
        if isinstance(city, str) and "chicago" in city.lower():
            key = name.lower().strip()
            restaurants_by_name[key].append({
                "id": match.get("id"),
                "name": name,
                "rating": meta.get("avg_rating", 0),
                "dishes": len(meta.get("menu_items", [])),
            })
    
    # Find duplicates and select best ones
    to_delete = []
    to_keep = []
    
    for name, entries in restaurants_by_name.items():
        if len(entries) > 1:
            # Sort by: dishes count (desc), rating (desc)
            sorted_entries = sorted(entries, key=lambda x: (x['dishes'], x['rating']), reverse=True)
            
            best = sorted_entries[0]
            rest = sorted_entries[1:]
            
            to_keep.append(best)
            to_delete.extend(rest)
            
            print(f"{best['name']} ({len(entries)} copies)")
            print(f"  ✅ Keeping: {best['id']} (rating: {best['rating']}, dishes: {best['dishes']})")
            for dup in rest:
                print(f"  ❌ Deleting: {dup['id']} (rating: {dup['rating']}, dishes: {dup['dishes']})")
            print()
    
    print(f"{'='*60}")
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
            print(f"   Remaining Chicago restaurants: {sum(len(v) for v in restaurants_by_name.values()) - len(to_delete)}")
        else:
            print("Aborted.")
    else:
        print("No duplicates to delete!")


if __name__ == "__main__":
    main()
