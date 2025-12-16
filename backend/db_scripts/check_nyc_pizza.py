"""Check NYC restaurants with pizza"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("menu-buddy")

# Get all restaurants
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
query_vec = model.encode("pizza restaurant").tolist()

results = index.query(
    vector=query_vec,
    top_k=10000,  # Increased to get all restaurants
    namespace="restaurants",
    include_metadata=True
)

# Filter for NYC restaurants with pizza
nyc_pizza = []
for m in results.matches:
    meta = m.metadata
    location = str(meta.get('location_json', '')).lower()
    menu_items = meta.get('menu_items', [])
    
    # Check if NYC location
    is_nyc = any(loc in location for loc in ['new york', 'brooklyn', 'manhattan', 'queens', 'bronx'])
    
    # Check if has pizza
    has_pizza = any('pizza' in str(item).lower() for item in menu_items)
    
    if is_nyc and has_pizza:
        pizza_items = [item for item in menu_items if 'pizza' in item.lower()]
        nyc_pizza.append({
            'name': meta.get('name'),
            'pizza_items': pizza_items[:3]
        })

print(f"NYC restaurants with pizza: {len(nyc_pizza)}\n")
for rest in nyc_pizza[:10]:
    print(f"  - {rest['name']}: {rest['pizza_items']}")
