# Yelp API → Menu Scraping → Pinecone Workflow

## Overview

This workflow uses the **Yelp Fusion API** (not web scraping) to get restaurant data, then scrapes the actual menu URLs to extract dishes, and finally uploads everything to Pinecone.

## Architecture

```
Yelp API → Menu URL → Dishes → Pinecone
   ↓          ↓          ↓         ↓
Business   menuUrl   Scrape    Upload
 Search    Extract    Menu     Vectors
```

## Files

1. **`yelp_api_client.py`** - Yelp API client
   - Searches businesses using Yelp Fusion API
   - Extracts `business.attributes.menuUrl`
   - Returns structured restaurant data

2. **`menu_url_scraper.py`** - Menu URL scraper
   - Scrapes HTML menus
   - Extracts dishes from PDF menus (using PyPDF2)
   - OCR for image menus (using pytesseract)
   - AI extraction with Groq

3. **`populate_from_yelp.py`** - Complete workflow
   - Orchestrates API → Scraping → Pinecone
   - Handles batch processing
   - Saves intermediate results

4. **`restaurant_to_pinecone.py`** - Pinecone uploader (existing)
   - Transforms restaurant data
   - Calculates taste vectors
   - Uploads to Pinecone

## Setup

### 1. Install Dependencies

```bash
pip install requests beautifulsoup4 pypdf2 pillow pytesseract sentence-transformers groq python-dotenv pinecone-client
```

### 2. Environment Variables

Add to your `.env` file:

```env
YELP_API_KEY=your_yelp_api_key_here
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=menu-buddy
```

Get Yelp API key: https://www.yelp.com/developers/v3/manage_app

### 3. Optional: Install Tesseract OCR

For image menu extraction:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Mac: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

## Usage

### Option 1: Search and Populate (Recommended)

Search for restaurants, scrape menus, and upload to Pinecone in one command:

```bash
cd backend
python populate_from_yelp.py --search "restaurants" --location "San Francisco" --limit 50
```

This will:
1. ✅ Search Yelp API for 50 restaurants in San Francisco
2. ✅ Extract menu URLs from business details
3. ✅ Scrape each menu URL for dishes
4. ✅ Calculate taste vectors
5. ✅ Upload to Pinecone

### Option 2: Process from JSON File

If you already have a JSON file with restaurant data:

```bash
python populate_from_yelp.py --file restaurants_with_menu_urls.json
```

### Option 3: Using Individual Scripts

#### Step 1: Get restaurants with menu URLs

```python
from yelp_api_client import YelpAPIClient

client = YelpAPIClient()
restaurants = client.search_and_get_menu_urls(
    term="restaurants",
    location="San Francisco",
    limit=50
)

# Save to file
import json
with open("restaurants.json", "w") as f:
    json.dump(restaurants, f, indent=2)
```

#### Step 2: Scrape menu URLs

```python
from menu_url_scraper import MenuURLScraper
import json

scraper = MenuURLScraper()

with open("restaurants.json", "r") as f:
    restaurants = json.load(f)

for restaurant in restaurants:
    menu_url = restaurant.get("menu_url")
    if menu_url:
        dishes = scraper.scrape_menu_url(menu_url)
        restaurant["menu_items"] = dishes
        restaurant["popular_dishes"] = dishes[:5]

# Save updated data
with open("restaurants_with_dishes.json", "w") as f:
    json.dump(restaurants, f, indent=2)
```

#### Step 3: Upload to Pinecone

```bash
python restaurant_to_pinecone.py restaurants_with_dishes.json
```

## Data Flow

### Input: Yelp API Response
```json
{
  "id": "business-id",
  "name": "Restaurant Name",
  "attributes": {
    "menu_url": "https://restaurant.com/menu"
  },
  "location": {...},
  "categories": [{"title": "Italian"}],
  "rating": 4.5
}
```

### After Menu Scraping
```json
{
  "id": "business-id",
  "name": "Restaurant Name",
  "menu_url": "https://restaurant.com/menu",
  "menu_items": [
    "Margherita Pizza",
    "Spaghetti Carbonara",
    "Caesar Salad"
  ],
  "location": {...},
  "cuisine_types": ["Italian"],
  "avg_rating": 4.5
}
```

### In Pinecone
```json
{
  "id": "restaurant:1",
  "values": [0.1, 0.2, ...],  // 384-dim embedding
  "metadata": {
    "name": "Restaurant Name",
    "menu_items": ["Margherita Pizza", ...],
    "taste_0": 0.2,  // sweet
    "taste_1": 0.3,  // salty
    "taste_2": 0.1,  // sour
    "taste_3": 0.0,  // bitter
    "taste_4": 0.4,  // umami
    "taste_5": 0.3   // spicy
  }
}
```

## Supported Menu Formats

### ✅ HTML Menus (Best Support)
- Most restaurant websites
- Detects menu sections automatically
- Falls back to AI extraction if needed

### ✅ PDF Menus
- Requires: `pip install pypdf2`
- Extracts text and uses AI to identify dishes
- Works with text-based PDFs (not scanned images)

### ✅ Image Menus
- Requires: `pip install pytesseract` + Tesseract OCR installation
- Uses OCR to extract text
- Uses AI to identify dishes from OCR text

## Troubleshooting

### No menu items extracted

If menu scraping fails:
1. Check if menu URL is accessible
2. Verify HTML structure (view source)
3. Ensure GROQ_API_KEY is set for AI extraction
4. Check console output for specific errors

### Yelp API errors

```bash
❌ Error: 401 Unauthorized
```
- Check your `YELP_API_KEY` in `.env`
- Verify key is active at https://www.yelp.com/developers/v3/manage_app

### Pinecone upload errors

```bash
❌ Error: PINECONE_API_KEY not found
```
- Add `PINECONE_API_KEY` to `.env`
- Verify index exists: `PINECONE_INDEX=menu-buddy`

## Differences from Old Scraper

| Old Approach | New Approach |
|-------------|--------------|
| ❌ Scrapes Yelp pages directly | ✅ Uses official Yelp API |
| ❌ Gets blocked with 403 | ✅ API access is authorized |
| ❌ Extracts categories as dishes | ✅ Scrapes actual menu URLs |
| ❌ "Pizza", "Italian" as items | ✅ "Margherita Pizza", "Carbonara" |

## Example Output

```bash
🚀 Starting workflow...

📡 STEP 1: Fetching restaurants from Yelp API...
   Found 50 restaurants
   Found 37 restaurants with menu URLs

🍽️  STEP 2: Scraping menu URLs for dishes...
[1/37] Joe's Pizza
   ✅ Found 12 dishes

[2/37] Thai House
   ✅ Found 24 dishes

...

📤 STEP 3: Uploading to Pinecone with taste vectors...
   Processing 37 restaurants with menus...
   Uploading 37 vectors to Pinecone...
   
✅ WORKFLOW COMPLETE!
   - Total restaurants found: 50
   - Restaurants with menu URLs: 37
   - Successfully scraped menus: 35
   - Uploaded to Pinecone: 35
```

## Next Steps

After populating Pinecone:

1. **Test queries**:
   ```bash
   python test_pinecone_query.py
   ```

2. **Verify data in frontend**:
   - Search for "I want pizza"
   - Should show actual dishes like "Margherita Pizza", not "Pizza", "Italian"

3. **Update taste vectors** (optional):
   ```bash
   python update_dish_taste_vectors.py
   ```
