# Embeddings & Taste Vectors Explained

## 🎯 Overview

Your Swaad system uses **TWO types of vectors** for restaurant and dish recommendations:

### 1. **Semantic Embeddings** (384 dimensions)
- **What**: Dense vector representations of text
- **Model**: SentenceTransformer "all-MiniLM-L6-v2"
- **Purpose**: Semantic search - finding restaurants based on meaning
- **Example**: "Thai restaurant" → finds restaurants with Thai cuisine

### 2. **Taste Vectors** (6 dimensions)
- **What**: Flavor profile representation
- **Dimensions**: `[sweet, salty, sour, bitter, umami, spicy]`
- **Purpose**: Matching user taste preferences with dishes
- **Example**: User likes spicy food → recommends dishes with high spicy score

---

## 📊 Current Data Structure in Pinecone

### **Namespace: `restaurants`** (100 vectors)

Each restaurant vector has:

```json
{
  "id": "restaurant:chalong",
  "values": [0.1, 0.2, ..., 0.3],  // 384-dim semantic embedding
  "metadata": {
    "name": "Chalong",
    "cuisine_types": ["Thai"],
    "avg_rating": 4.6,
    "price_range": 3,
    "menu_items": ["Pad Thai", "Green Curry", "Tom Yum"],  // Just strings (OLD)
    "taste_vector": null  // Restaurant-level taste (currently null)
  }
}
```

### **Namespace: `ingredients`** (500 vectors)

Ingredient taste profiles for semantic taste inference.

---

## ❌ Current Problems

### Problem 1: No Dish-Level Taste Vectors
- Menu items are just strings: `["Pad Thai", "Green Curry"]`
- Taste vectors calculated **on-the-fly** for every request
- **Slow** and **inconsistent**

### Problem 2: Restaurant Taste Vector is Null
- `taste_vector: null` in metadata
- Should be average of all dish taste vectors

### Problem 3: Inefficient Computation
- Recalculating taste vectors for 20 dishes × 10 restaurants = 200 calculations per query
- Should be pre-calculated and stored

---

## ✅ Recommended Solution

### **Update Data Structure:**

```json
{
  "id": "restaurant:chalong",
  "values": [0.1, 0.2, ..., 0.3],  // 384-dim semantic embedding (unchanged)
  "metadata": {
    "name": "Chalong",
    "cuisine_types": ["Thai"],
    "avg_rating": 4.6,
    "price_range": 3,
    
    // NEW: Dishes with pre-calculated taste vectors
    "dishes": [
      {
        "name": "Pad Thai",
        "taste": [0.3, 0.4, 0.1, 0.0, 0.5, 0.2]  // [sweet, salty, sour, bitter, umami, spicy]
      },
      {
        "name": "Green Curry",
        "taste": [0.2, 0.3, 0.1, 0.0, 0.4, 0.7]
      }
    ],
    
    // NEW: Restaurant-level taste (average of all dishes)
    "taste_vector": [0.25, 0.35, 0.1, 0.0, 0.45, 0.45]
  }
}
```

---

## 🚀 How to Update

Run the update script:

```bash
cd backend
python3.10 update_dish_taste_vectors.py
```

This will:
1. ✅ Fetch all 100 restaurants from Pinecone
2. ✅ Calculate taste vectors for each dish
3. ✅ Calculate restaurant-level taste vector (average)
4. ✅ Update Pinecone with new structure

**Time**: ~2-3 minutes for 100 restaurants

---

## 📈 Benefits After Update

### Before:
- ❌ 200 taste calculations per query
- ❌ ~500ms query time
- ❌ Inconsistent results

### After:
- ✅ 0 taste calculations (pre-computed)
- ✅ ~50ms query time (10x faster)
- ✅ Consistent results
- ✅ Can filter by taste profile
- ✅ Better dish recommendations

---

## 🔍 How It Works

### Query Flow:

1. **User Query**: "I want spicy Thai food"
   
2. **Semantic Search** (384-dim embedding):
   ```
   Query embedding → Pinecone → Top 10 Thai restaurants
   ```

3. **Taste Ranking** (6-dim taste vector):
   ```
   User taste: [0.1, 0.2, 0.1, 0.0, 0.3, 0.9]  // High spicy (0.9)
   Restaurant taste: [0.25, 0.35, 0.1, 0.0, 0.45, 0.7]  // High spicy (0.7)
   Similarity: 0.85 ✅
   ```

4. **Dish Recommendations**:
   ```
   For each dish in restaurant:
     Calculate similarity(user_taste, dish_taste)
     Return top 5 dishes
   ```

---

## 🎓 Key Concepts

### Semantic Embedding
- Captures **meaning** and **context**
- "Thai restaurant" is similar to "Bangkok cuisine"
- Used for initial search

### Taste Vector
- Captures **flavor profile**
- Independent of language/culture
- Used for ranking and personalization

### Why Both?
- **Semantic**: Find relevant restaurants
- **Taste**: Rank by user preference
- **Combined**: Best of both worlds!

---

## 📝 Next Steps

1. ✅ Run `update_dish_taste_vectors.py` to update Pinecone
2. ✅ Test queries to see improved dish recommendations
3. ✅ Monitor query performance (should be 10x faster)
4. ✅ Consider adding more metadata (price, dietary info, etc.)

