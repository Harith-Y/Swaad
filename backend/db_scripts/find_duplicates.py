"""
Check for duplicate Chicago restaurants in Pinecone
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index


def main():
    """Find duplicate restaurants"""
    
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
                "rating": meta.get("avg_rating"),
                "dishes": len(meta.get("menu_items", [])),
                "taste_vector": [
                    meta.get("taste_0", 0),
                    meta.get("taste_1", 0),
                    meta.get("taste_2", 0),
                    meta.get("taste_3", 0),
                    meta.get("taste_4", 0),
                    meta.get("taste_5", 0)
                ]
            })
    
    # Find duplicates
    duplicates = {name: entries for name, entries in restaurants_by_name.items() if len(entries) > 1}
    
    print(f"Total Chicago restaurants: {sum(len(v) for v in restaurants_by_name.values())}")
    print(f"Unique names: {len(restaurants_by_name)}")
    print(f"Duplicates found: {len(duplicates)}\n")
    
    if duplicates:
        print("="*60)
        print("DUPLICATE RESTAURANTS:")
        print("="*60)
        
        for name, entries in sorted(duplicates.items()):
            print(f"\n{entries[0]['name']} ({len(entries)} copies)")
            for i, entry in enumerate(entries):
                taste_sum = sum(entry['taste_vector'])
                print(f"  [{i+1}] ID: {entry['id']}")
                print(f"      Rating: {entry['rating']}, Dishes: {entry['dishes']}")
                print(f"      Taste: {taste_sum:.2f} (sum of vector)")
        
        print(f"\n{'='*60}")
        print(f"Total duplicate entries: {sum(len(v) - 1 for v in duplicates.values())}")
        print("These should be removed to avoid duplicate results.")
    else:
        print("✅ No duplicates found!")


if __name__ == "__main__":
    main()
