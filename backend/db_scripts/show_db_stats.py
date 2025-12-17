"""
Show database statistics after cleanup
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index


def main():
    """Show final database stats"""
    
    print("Fetching database statistics...\n")
    
    pc_index = get_pinecone_index()
    
    # Get index stats
    stats = pc_index.describe_index_stats()
    total = stats.get('total_vector_count', 0)
    
    print(f"{'='*60}")
    print(f"DATABASE STATISTICS")
    print(f"{'='*60}\n")
    print(f"Total vectors: {total}")
    print(f"Namespaces: {stats.get('namespaces', {})}\n")
    
    # Query for all restaurants
    dummy_vector = [0.0] * 384
    results = pc_index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        namespace="restaurants"
    )
    
    restaurants = results.get('matches', [])
    print(f"Total restaurants: {len(restaurants)}\n")
    
    # Stats by city
    city_stats = defaultdict(lambda: {'total': 0, 'with_menus': 0, 'total_dishes': 0})
    
    for match in restaurants:
        meta = match.get('metadata', {})
        city = meta.get('city', 'Unknown')
        menu_items = meta.get('menu_items', [])
        
        city_stats[city]['total'] += 1
        if menu_items:
            city_stats[city]['with_menus'] += 1
            city_stats[city]['total_dishes'] += len(menu_items)
    
    print(f"{'='*60}")
    print("RESTAURANTS BY CITY")
    print(f"{'='*60}\n")
    
    for city in sorted(city_stats.keys()):
        stats = city_stats[city]
        avg_dishes = stats['total_dishes'] / stats['with_menus'] if stats['with_menus'] > 0 else 0
        print(f"{city}:")
        print(f"  Total restaurants: {stats['total']}")
        print(f"  With menus: {stats['with_menus']}")
        print(f"  Total dishes: {stats['total_dishes']}")
        print(f"  Avg dishes per restaurant: {avg_dishes:.1f}")
        print()
    
    # Overall stats
    total_with_menus = sum(s['with_menus'] for s in city_stats.values())
    total_dishes = sum(s['total_dishes'] for s in city_stats.values())
    overall_avg = total_dishes / total_with_menus if total_with_menus > 0 else 0
    
    print(f"{'='*60}")
    print(f"OVERALL")
    print(f"{'='*60}")
    print(f"Total restaurants: {len(restaurants)}")
    print(f"Restaurants with menus: {total_with_menus}")
    print(f"Total dishes: {total_dishes}")
    print(f"Average dishes per restaurant: {overall_avg:.1f}")
    print()
    
    print("✅ Database cleanup complete!")


if __name__ == "__main__":
    main()
