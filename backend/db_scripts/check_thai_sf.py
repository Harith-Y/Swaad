"""
Check Thai restaurants in San Francisco
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index


def check_thai_sf():
    """Check for Thai restaurants in San Francisco"""
    print("Searching for Thai restaurants in San Francisco...\n")
    
    try:
        pc_index = get_pinecone_index()
        
        # Query for all restaurants
        dummy_vector = [0.0] * 384
        
        results = pc_index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True,
            namespace="restaurants"
        )
        
        # Filter for SF Thai restaurants
        sf_thai = []
        all_sf = []
        
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            location = str(meta.get("location", "")).lower()
            cuisine_types = meta.get("cuisine_types", [])
            
            # Convert cuisine_types to list if string
            if isinstance(cuisine_types, str):
                import json
                try:
                    cuisine_types = json.loads(cuisine_types)
                except:
                    cuisine_types = [cuisine_types]
            
            cuisine_str = " ".join([str(c).lower() for c in cuisine_types])
            
            # Check if SF
            is_sf = any(x in location for x in ["san francisco", "sf,"])
            
            if is_sf:
                all_sf.append({
                    "name": meta.get("name"),
                    "location": location,
                    "cuisine": cuisine_types,
                    "dishes": len(meta.get("menu_items", []))
                })
                
                # Check if Thai
                if "thai" in cuisine_str:
                    sf_thai.append({
                        "name": meta.get("name"),
                        "location": location,
                        "cuisine": cuisine_types,
                        "dishes": len(meta.get("menu_items", [])),
                        "menu_items": meta.get("menu_items", [])
                    })
        
        print(f"📊 San Francisco Statistics:")
        print(f"   Total SF restaurants: {len(all_sf)}")
        print(f"   Thai restaurants: {len(sf_thai)}\n")
        
        if sf_thai:
            print(f"✅ Found {len(sf_thai)} Thai restaurant(s) in San Francisco:\n")
            for rest in sf_thai:
                print(f"  🍽️  {rest['name']}")
                print(f"     Location: {rest['location']}")
                print(f"     Cuisine: {rest['cuisine']}")
                print(f"     Dishes: {rest['dishes']}")
                
                # Check for curry dishes
                curry_dishes = [d for d in rest['menu_items'] if 'curry' in d.lower()]
                if curry_dishes:
                    print(f"     🍛 Curry dishes: {len(curry_dishes)}")
                    for dish in curry_dishes[:5]:
                        print(f"        - {dish}")
                else:
                    print(f"     🍛 Curry dishes: None")
                print()
        else:
            print("❌ No Thai restaurants found in San Francisco")
            print("\nSample SF restaurants:")
            for rest in all_sf[:10]:
                print(f"  - {rest['name']} ({', '.join(rest['cuisine'])})")
        
        return len(sf_thai)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    check_thai_sf()
