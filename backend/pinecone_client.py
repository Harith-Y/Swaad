"""
Pinecone vector database client and operations.
"""
from typing import Optional, List, Dict, Any
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX
import csv
from pathlib import Path
from embeddings import embed_text, get_embedding_model


# Global Pinecone index instance
_pinecone_index = None
_ingredient_upsert_done = False


def get_pinecone_index():
    """Get or initialize Pinecone index."""
    global _pinecone_index
    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX not in existing_indexes:
            # Create index if it doesn't exist
            from pinecone import ServerlessSpec
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        _pinecone_index = pc.Index(PINECONE_INDEX)
    
    return _pinecone_index


def query_pinecone(
    query_vector: List[float],
    top_k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True
) -> List[Dict[str, Any]]:
    """Query Pinecone index for similar vectors."""
    index = get_pinecone_index()
    
    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": include_metadata
    }
    
    if filter_dict:
        query_params["filter"] = filter_dict
    
    result = index.query(**query_params)
    
    # Convert to list of dicts
    matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
    return matches


def upsert_to_pinecone(vectors: List[Dict[str, Any]]) -> None:
    """Upsert vectors to Pinecone index."""
    index = get_pinecone_index()
    index.upsert(vectors=vectors)


def maybe_upsert_ingredients_to_pinecone() -> None:
    """
    Upsert ingredient flavor data to Pinecone if not already done.
    This is called once on startup to ensure ingredient data is available.
    """
    global _ingredient_upsert_done
    
    if _ingredient_upsert_done:
        return
    
    csv_path = Path("ingredient-flavor.csv")
    if not csv_path.exists():
        print(f"[WARNING] {csv_path} not found, skipping ingredient upsert")
        _ingredient_upsert_done = True
        return
    
    try:
        index = get_pinecone_index()
        model = get_embedding_model()
        
        # Check if ingredients already exist
        stats = index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0) if isinstance(stats, dict) else getattr(stats, "total_vector_count", 0)
        
        if total_vectors > 100:  # Assume ingredients are already loaded
            print(f"[INFO] Pinecone already has {total_vectors} vectors, skipping ingredient upsert")
            _ingredient_upsert_done = True
            return
        
        # Load and upsert ingredients
        vectors = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ingredient = row.get("ingredient", "").strip()
                if not ingredient:
                    continue
                
                # Create embedding
                embedding = model.encode(ingredient).tolist()
                
                # Create metadata
                metadata = {
                    "type": "ingredient",
                    "name": ingredient,
                    "sweet": float(row.get("sweet", 0)),
                    "salty": float(row.get("salty", 0)),
                    "sour": float(row.get("sour", 0)),
                    "bitter": float(row.get("bitter", 0)),
                    "umami": float(row.get("umami", 0)),
                    "spicy": float(row.get("spicy", 0)),
                }
                
                vectors.append({
                    "id": f"ingredient:{ingredient}",
                    "values": embedding,
                    "metadata": metadata
                })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        print(f"[INFO] Upserted {len(vectors)} ingredients to Pinecone")
        _ingredient_upsert_done = True
        
    except Exception as e:
        print(f"[ERROR] Failed to upsert ingredients: {e}")
        _ingredient_upsert_done = True  # Don't retry on every request

