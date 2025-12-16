# Restaurant Dish Population Scripts

This directory contains comprehensive scripts for populating restaurant dish data into Pinecone from Yelp API.

## 📋 Available Scripts

### 1. **master_populate.py** (Recommended)
**The most comprehensive solution** - Uses multiple search strategies to find and populate restaurants.

```bash
python master_populate.py
```

**Features:**
- Searches 15+ categories per location (pizza, italian, chinese, etc.)
- Intelligent deduplication
- Automatic menu scraping
- Comprehensive error handling
- Detailed progress tracking and final report
- Targets: SF, San Jose, Oakland, NYC, Brooklyn, Manhattan

**Expected Results:**
- 500-1000+ unique restaurants
- High coverage across cuisines
- Detailed dish data

---

### 2. **populate_pizza_restaurants.py** (Pizza Focus)
Specifically targets pizza restaurants for better pizza search results.

```bash
python populate_pizza_restaurants.py
```

**Features:**
- Focuses on pizza/pizzeria searches
- Multiple NYC boroughs
- Bay Area cities
- Ensures good pizza dish coverage

---

### 3. **populate_dishes_comprehensive.py**
Basic comprehensive population with customizable parameters.

```bash
python populate_dishes_comprehensive.py --locations "New York, NY" "San Francisco, CA" --limit 100
```

**Options:**
- `--locations`: List of locations to populate
- `--limit`: Number of restaurants per location
- `--category`: Search category (default: "restaurants")

---

### 4. **populate_with_yelp_ai.py**
Enhanced search using multiple search terms per location.

```bash
python populate_with_yelp_ai.py --limit 100
```

**Features:**
- Multiple search terms (pizza, italian, asian, etc.)
- Enhanced data collection
- Better menu URL detection

---

### 5. **populate_from_yelp.py** (Legacy)
Original population script - single search term.

```bash
python populate_from_yelp.py --search "restaurants" --location "New York, NY" --limit 50
```

---

## 🚀 Quick Start (Recommended)

For the best results, run the master script:

```bash
cd backend/db_scripts
python master_populate.py
```

This will:
1. Search 6 locations (SF, SJ, Oakland, NYC, Brooklyn, Manhattan)
2. Use 15 different search categories per location
3. Find 500-1000+ unique restaurants
4. Scrape menus and extract dishes
5. Upload to Pinecone with proper metadata

**Estimated time:** 2-3 hours for full population

---

## 🍕 Quick Pizza Population

If you specifically need more pizza restaurants:

```bash
python populate_pizza_restaurants.py
```

This focuses only on pizza places and will complete in ~30 minutes.

---

## 📊 Checking Results

After population, verify the data:

```bash
# Check total restaurants
python check_multi_city_restaurants.py

# Check pizza specifically
python check_pizza.py

# Check NYC pizza
python check_nyc_pizza.py

# Quick overview
python quick_pinecone_check.py
```

---

## 🔧 How It Works

1. **Search**: Uses Yelp Fusion API to search restaurants by category/cuisine
2. **Enhance**: Gets detailed business info including menu URLs
3. **Scrape**: Extracts dish names from menu URLs (PDF, HTML, images via OCR)
4. **Process**: Generates taste vectors for each dish using AI
5. **Upload**: Stores in Pinecone with proper metadata for semantic search

---

## 📝 Data Structure

Each restaurant in Pinecone contains:

```json
{
  "name": "Restaurant Name",
  "location_json": {"city": "San Francisco", "state": "CA", ...},
  "cuisine_types": ["Italian", "Pizza"],
  "avg_rating": 4.5,
  "price_range": "$$",
  "menu_items": ["Margherita Pizza", "Pepperoni Pizza", ...],
  "popular_dishes": ["Top 10 dishes"],
  "menu_url": "https://...",
  "phone": "...",
  "hours": [...]
}
```

---

## 🎯 Target Locations

- **Bay Area**: San Francisco, San Jose, Oakland
- **NYC Metro**: New York, Brooklyn, Manhattan

These locations have regional grouping - users in SF will see results from SJ and Oakland too.

---

## ⚠️ Common Issues

### Issue: Few dishes found
**Solution:** Run `master_populate.py` which uses multiple search strategies

### Issue: Menu scraping fails
**Solution:** Check menu URL format - some sites block scrapers. The script handles this gracefully.

### Issue: Rate limiting
**Solution:** Scripts have built-in delays. If you hit limits, wait 24 hours or get a higher-tier Yelp API key.

### Issue: Duplicates
**Solution:** Scripts use intelligent deduplication by business ID

---

## 📈 Expected Results

After running `master_populate.py`:

- **Restaurants**: 500-1000+ unique
- **Dishes**: 10,000-30,000+ 
- **Pizza places**: 50-100+
- **Coverage**: 15+ cuisines per city

---

## 🔍 Verification

```bash
# Total count
python -c "from pinecone import Pinecone; import os; from dotenv import load_dotenv; load_dotenv(); pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY')); idx = pc.Index('menu-buddy'); print(idx.describe_index_stats())"

# NYC pizza count
python check_nyc_pizza.py

# Multi-city stats
python check_multi_city_restaurants.py
```

---

## 🎉 Success Criteria

Your population is successful when:

✅ 500+ restaurants in Pinecone  
✅ 50+ pizza restaurants  
✅ Restaurants have 10+ dishes on average  
✅ All 6 cities represented  
✅ "I want pizza" query returns 5+ results per city

---

## 💡 Tips

1. **Run overnight**: Full population takes 2-3 hours
2. **Start with pizza**: Quick validation in 30 min
3. **Check frequently**: Use verification scripts
4. **Monitor errors**: Scripts log all issues
5. **Save results**: JSON files saved for backup

---

## 🆘 Support

If population fails:
1. Check `.env` has `YELP_API_KEY`
2. Verify Yelp API quota (50,000 calls/day)
3. Check internet connection
4. Review error logs in console
5. Try smaller `--limit` values

---

## 📚 Related Files

- `yelp_api_client.py` - Yelp API wrapper
- `menu_url_scraper.py` - Menu extraction logic
- `restaurant_to_pinecone.py` - Pinecone upload logic
- `dish_processing.py` - Location matching, taste inference

---

**Last Updated:** December 16, 2025
