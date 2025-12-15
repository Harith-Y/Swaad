"""Quick check of what's in Pinecone"""
import dotenv
from pinecone import Pinecone

PINECONE_API_KEY = dotenv.get_key(".env", "PINECONE_API_KEY")
PINECONE_INDEX = "menu-buddy"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Get stats
stats = index.describe_index_stats()
print(f"Total vectors in Pinecone: {stats.get('total_vector_count', 0)}")
print(f"Namespaces: {stats.get('namespaces', {})}")

# Query for restaurants with "pizza" in menu
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
query_vec = model.encode("pizza restaurant").tolist()

results = index.query(
    vector=query_vec,
    top_k=100,
    include_metadata=True,
    namespace="restaurants"
)

matches = results.get("matches", [])
print(f"\nTotal restaurants returned: {len(matches)}")

# Count how many have "pizza" in menu
pizza_count = 0
for m in matches:
    meta = m.get("metadata", {})
    menu_items = meta.get("menu_items", [])
    has_pizza = any("pizza" in item.lower() for item in menu_items)
    if has_pizza:
        pizza_count += 1
        print(f"\n✅ {meta.get('name')} ({meta.get('location_json', 'Unknown')})")
        pizza_dishes = [item for item in menu_items if "pizza" in item.lower()]
        print(f"   Pizza dishes: {', '.join(pizza_dishes[:5])}")

print(f"\n📊 Summary: {pizza_count} out of {len(matches)} restaurants have pizza in their menu")
