# Quick Start Guide - Restaurant Population

## Step 1: Ensure Requirements

Make sure you have all required packages:

```bash
cd c:\Users\yhari\Documents\Coding\Intern\Swaad-ne\backend
pip install requests beautifulsoup4 pypdf2 pillow sentence-transformers groq python-dotenv pinecone-client
```

## Step 2: Verify Environment

Ensure your `.env` file has:
```
YELP_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=menu-buddy
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

## Step 3: Run Population

### Option A: Full Population (Recommended - takes 2-3 hours)
```bash
cd backend\db_scripts
python master_populate.py
```

This will populate 500-1000+ restaurants across all 6 locations.

### Option B: Quick Pizza Population (30 minutes)
```bash
cd backend\db_scripts
python populate_pizza_restaurants.py
```

This focuses on pizza restaurants only - great for quick testing.

### Option C: Single Location (Quick Test - 15 minutes)
```bash
cd backend\db_scripts
python populate_from_yelp.py --search "restaurants" --location "New York, NY" --limit 100
```

## Step 4: Verify Results

```bash
cd backend\db_scripts
python check_multi_city_restaurants.py
python check_nyc_pizza.py
```

## What Each Script Does

| Script | Time | Restaurants | Best For |
|--------|------|-------------|----------|
| `master_populate.py` | 2-3h | 500-1000+ | Production, full coverage |
| `populate_pizza_restaurants.py` | 30m | 50-100 pizza | Pizza testing |
| `populate_from_yelp.py` | 15m | 50-100 | Quick testing, single location |

## Monitoring Progress

The scripts will show real-time progress:
```
📍 COMPREHENSIVE SEARCH: New York, NY
🔍 Searching: 'pizza restaurants' in New York, NY
   Found 50 results, 35 new
[1/35] Joe's Pizza
   🍽️  Scraping menu...
   ✅ Found 25 dishes
```

## Expected Final Results

After `master_populate.py`:
- 🏪 500-1000 restaurants
- 🍕 50-100 pizza places  
- 🍽️ 10,000-30,000 dishes
- 📍 6 cities covered

## Troubleshooting

**ModuleNotFoundError:** Run `pip install -r requirements.txt` first

**Yelp API Error:** Check your API key in `.env`

**Rate Limit:** Wait 24 hours or reduce `--limit` values

**No dishes found:** Menu URLs may be blocked - script will skip and continue

## Quick Commands

```powershell
# Activate venv
.\\venv\\Scripts\\Activate.ps1

# Install requirements
pip install requests beautifulsoup4 pypdf2 pillow sentence-transformers groq python-dotenv pinecone-client

# Run master populate
cd backend\\db_scripts
python master_populate.py

# Check results
python check_multi_city_restaurants.py
```

## Success Indicators

✅ See "✅ Successfully uploaded X restaurants" messages  
✅ check_multi_city_restaurants.py shows 500+ total  
✅ "I want pizza" query returns multiple results  
✅ JSON files saved with restaurant data

---

**Ready to start?** Just run:
```powershell
cd c:\\Users\\yhari\\Documents\\Coding\\Intern\\Swaad-ne\\backend\\db_scripts
python master_populate.py
```
