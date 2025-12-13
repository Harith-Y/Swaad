# Swaad Backend - Modular Architecture

## 📁 Project Structure

```
backend/
├── config.py                    # Configuration and environment variables (39 lines)
├── database.py                  # User data management (142 lines)
├── dish_processing.py           # Dish extraction and classification (155 lines)
├── embeddings.py                # Vector embeddings operations (62 lines)
├── main.py                      # FastAPI app initialization (63 lines)
├── models.py                    # Pydantic data models (93 lines)
├── pinecone_client.py           # Pinecone vector database client (145 lines)
├── recommendations.py           # Recommendation logic (208 lines)
├── taste_analysis.py            # Taste vector calculations (159 lines)
├── routes/
│   ├── __init__.py
│   └── chat.py                  # Chat endpoint handler (176 lines)
├── restaurant_to_pinecone.py    # Data upload script
└── test_pinecone_upload.py      # Testing utilities
```

**Total modular code: 1,242 lines** (down from 1,802 lines in monolithic version)

---

## 🎯 Module Descriptions

### **config.py**
Central configuration management for the entire application.

**Responsibilities:**
- Load environment variables
- Define API keys (GROQ, Pinecone)
- Set model configurations
- Feature flags (semantic taste inference)
- Default settings and constants

**Key exports:**
- `GROQ_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX`
- `SENTENCE_TRANSFORMER_MODEL`
- `USE_SEMANTIC_DISH_TASTE`, `USE_SEMANTIC_INGREDIENT_TASTE`
- `TASTE_DIMENSIONS`, `TASTE_VECTOR_SIZE`

---

### **models.py**
Pydantic models for request/response validation and data structures.

**Key models:**
- `ChatRequest` - Chat endpoint input
- `UserProfile` - User preferences and dietary info
- `DishInput` - Dish name and category
- `RestaurantRecommendation` - Restaurant data structure
- `ChatResponse` - Chat endpoint output

---

### **database.py**
User data management and in-memory storage.

**Responsibilities:**
- Manage dummy user data (replace with real DB in production)
- User profile CRUD operations
- Normalize and validate user preferences
- Sync user metadata from requests

**Key functions:**
- `get_dummy_user(user_key)` - Retrieve user data
- `update_dummy_user(user_key, data)` - Update user preferences
- `dummy_user_to_user_profile(user)` - Convert to Pydantic model
- `sync_dummy_user_from_request(request)` - Sync from API request

---

### **embeddings.py**
Vector embedding operations using SentenceTransformer.

**Responsibilities:**
- Initialize and manage SentenceTransformer model
- Generate text embeddings
- Calculate cosine similarity
- Vector combination and normalization

**Key functions:**
- `get_embedding_model()` - Get/initialize model
- `embed_text(text)` - Generate embedding vector
- `calculate_cosine_similarity(vec1, vec2)` - Similarity score
- `combine_vectors(primary, secondary, weight)` - Weighted average

---

### **pinecone_client.py**
Pinecone vector database client and operations.

**Responsibilities:**
- Initialize Pinecone connection
- Query vector database
- Upsert vectors (restaurants, ingredients)
- Manage ingredient flavor data

**Key functions:**
- `get_pinecone_index()` - Get/initialize Pinecone index
- `query_pinecone(vector, top_k, filter)` - Semantic search
- `upsert_to_pinecone(vectors)` - Upload vectors
- `maybe_upsert_ingredients_to_pinecone()` - Load ingredient data

---

### **taste_analysis.py**
Taste vector analysis and similarity calculations.

**Responsibilities:**
- Load ingredient flavor mappings from CSV
- Infer taste vectors from text (keyword or semantic)
- Calculate taste similarity scores
- Convert user profiles to taste vectors

**Key functions:**
- `load_ingredient_flavor_map()` - Load CSV data
- `infer_taste_from_text(text)` - Keyword-based inference
- `infer_taste_from_text_semantic(text)` - Semantic inference via Pinecone
- `user_profile_to_taste_vector(profile)` - User taste preferences
- `taste_similarity(user_vec, item_vec)` - Similarity score
- `favorites_boost(menu, favorites)` - Boost score for favorites

**Taste Vector Format:** `[sweet, salty, sour, bitter, umami, spicy]` (6D)

---

### **dish_processing.py**
Dish extraction, classification, and filtering utilities.

**Responsibilities:**
- Initialize Groq AI client
- Classify dishes by category (appetizer/mains/desserts)
- Filter dishes by dietary preferences (veg/non-veg)
- Check for allergens
- Normalize dish names

**Key functions:**
- `get_groq_client()` - Get/initialize Groq client
- `classify_dish_with_groq(dish)` - AI-based classification
- `filter_dishes_by_diet(dishes, diet_type)` - Diet filtering
- `allergy_filter(menu, allergies)` - Allergen detection
- `is_nonveg_text(text)` - Non-veg keyword detection
- `normalize_dish_name(name)` - Name normalization

---

### **recommendations.py**
Restaurant and dish recommendation logic.

**Responsibilities:**
- Rank restaurants by combined score
- Generate dish recommendations for each restaurant
- Filter by diet and allergies
- Calculate aggregate taste vectors

**Key functions:**
- `dish_recommendations_for_restaurant(menu, user_taste, diet)` - Top dishes
- `rank_restaurants(restaurants, user_taste, favorites)` - Ranking logic
- `filter_and_rank_recommendations(matches, ...)` - Complete pipeline
- `calculate_restaurant_taste_vector(menu)` - Aggregate taste

**Scoring Formula:**
```
combined_score = semantic_score + 0.35 * taste_similarity + favorites_boost
```

---

### **routes/chat.py**
Chat endpoint implementation for restaurant recommendations.

**Responsibilities:**
- Handle chat requests
- Query Pinecone for semantic search
- Apply taste-based ranking
- Rewrite AI responses for diet preferences
- Manage conversation state

**Main endpoint:**
- `chat_endpoint(request: ChatRequest)` - Async handler

**Flow:**
1. Load user preferences
2. Calculate user taste vector
3. Query Pinecone for restaurants
4. Filter by diet and allergies
5. Rank by combined score
6. Generate dish recommendations
7. Rewrite response text with Groq AI

---

### **main.py**
FastAPI application initialization and routing.

**Responsibilities:**
- Initialize FastAPI app
- Configure CORS
- Preload models on startup
- Register API endpoints

**Endpoints:**
- `GET /` - Health check
- `POST /api/chat` - Restaurant recommendations

---

## 🚀 Running the Server

```bash
cd backend
python3.10 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📊 Benefits of Modular Architecture

1. **Maintainability** - Each module has a single responsibility
2. **Testability** - Easy to unit test individual modules
3. **Readability** - Clear separation of concerns
4. **Scalability** - Easy to add new features
5. **Reusability** - Modules can be imported anywhere
6. **Debugging** - Easier to locate and fix issues

---

## 🔄 Migration Notes

- **Old file:** `main_old.py` (1,802 lines)
- **New structure:** 10 modules (1,242 lines total)
- **Reduction:** 31% smaller, much more organized
- **Backward compatible:** Same API endpoints and behavior

