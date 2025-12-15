"""
Menu URL Scraper - Extracts dishes from menu URLs (HTML/PDF/Image)

This script scrapes actual menu URLs (not Yelp pages) to extract dish names.
Supports:
- HTML menus (most restaurant websites)
- PDF menus (using PyPDF2 or pdfplumber)
- Image menus (using OCR with pytesseract/EasyOCR)

Requirements:
    pip install requests beautifulsoup4 pypdf2 pillow pytesseract groq python-dotenv

Usage:
    scraper = MenuURLScraper()
    dishes = scraper.scrape_menu_url("https://restaurant.com/menu")
"""

import os
import re
import requests
from typing import List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:
    print("⚠️  Groq not installed. AI extraction will not be available.")
    Groq = None


class MenuURLScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        
        # Initialize Groq for AI extraction
        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key) if groq_key and Groq else None
        
        # Category words to filter out
        self.category_words = {
            'appetizers', 'entrees', 'desserts', 'beverages', 'drinks', 'sides',
            'starters', 'mains', 'salads', 'soups', 'seafood', 'vegetarian',
            'lunch', 'dinner', 'breakfast', 'brunch', 'specials', 'menu',
            'italian', 'chinese', 'thai', 'indian', 'japanese', 'mexican',
            'american', 'french', 'korean', 'vietnamese', 'mediterranean',
            'pizza', 'pasta', 'sushi', 'burgers', 'sandwiches', 'wings',
            'noodles', 'rice', 'curry', 'bbq', 'grill', 'steakhouse'
        }
    
    def scrape_menu_url(self, menu_url: str) -> List[str]:
        """
        Scrape dishes from a menu URL
        
        Args:
            menu_url: URL to the restaurant's menu (HTML/PDF/Image)
        
        Returns:
            List of dish names
        """
        if not menu_url:
            return []
        
        print(f"🔍 Scraping menu: {menu_url}")
        
        # Determine URL type and route to appropriate handler
        url_lower = menu_url.lower()
        
        if url_lower.endswith('.pdf'):
            return self._scrape_pdf_menu(menu_url)
        elif any(url_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            return self._scrape_image_menu(menu_url)
        else:
            return self._scrape_html_menu(menu_url)
    
    def _scrape_html_menu(self, url: str) -> List[str]:
        """Scrape dishes from HTML menu page"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            dishes = []
            
            # Strategy 1: Look for common menu item selectors
            menu_selectors = [
                'div.menu-item',
                'li.menu-item',
                'div.dish',
                'div.food-item',
                'article.menu-item',
                'div[class*="menu"]',
                'div[class*="dish"]',
                'div[class*="item"]'
            ]
            
            for selector in menu_selectors:
                items = soup.select(selector)
                if items:
                    for item in items:
                        # Look for dish name in headings or strong tags
                        name_elem = item.find(['h3', 'h4', 'h5', 'strong', 'span'])
                        if name_elem:
                            dish_name = name_elem.get_text(strip=True)
                            if self._is_valid_dish_name(dish_name):
                                dishes.append(dish_name)
            
            # Strategy 2: Look for headings that might be dish names
            if len(dishes) < 5:
                headings = soup.find_all(['h2', 'h3', 'h4', 'h5'])
                for heading in headings:
                    text = heading.get_text(strip=True)
                    if self._is_valid_dish_name(text):
                        dishes.append(text)
            
            # Strategy 3: Use AI extraction if available and other methods failed
            if len(dishes) < 5 and self.groq_client:
                print("   Using AI to extract dishes from HTML...")
                text_content = soup.get_text(separator=' ', strip=True)[:4000]  # Limit text
                ai_dishes = self._extract_dishes_with_ai(text_content)
                dishes.extend(ai_dishes)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_dishes = []
            for dish in dishes:
                dish_lower = dish.lower()
                if dish_lower not in seen:
                    seen.add(dish_lower)
                    unique_dishes.append(dish)
            
            print(f"   ✅ Extracted {len(unique_dishes)} dishes from HTML")
            return unique_dishes[:30]  # Limit to 30 dishes
        
        except Exception as e:
            print(f"   ❌ Error scraping HTML menu: {e}")
            return []
    
    def _scrape_pdf_menu(self, url: str) -> List[str]:
        """Scrape dishes from PDF menu"""
        print("   📄 PDF menu detected")
        
        try:
            # Download PDF
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Try using PyPDF2
            try:
                import PyPDF2
                from io import BytesIO
                
                pdf_file = BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                # Extract dishes from text using AI
                if self.groq_client and text:
                    print("   Using AI to extract dishes from PDF...")
                    dishes = self._extract_dishes_with_ai(text[:4000])
                    print(f"   ✅ Extracted {len(dishes)} dishes from PDF")
                    return dishes
                else:
                    print("   ⚠️  No AI client available for PDF extraction")
                    return []
            
            except ImportError:
                print("   ⚠️  PyPDF2 not installed. Install with: pip install PyPDF2")
                return []
        
        except Exception as e:
            print(f"   ❌ Error scraping PDF menu: {e}")
            return []
    
    def _scrape_image_menu(self, url: str) -> List[str]:
        """Scrape dishes from image menu using OCR"""
        print("   🖼️  Image menu detected")
        
        try:
            # Download image
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Try using pytesseract
            try:
                import pytesseract
                from PIL import Image
                from io import BytesIO
                
                image = Image.open(BytesIO(response.content))
                text = pytesseract.image_to_string(image)
                
                # Extract dishes from OCR text using AI
                if self.groq_client and text:
                    print("   Using AI to extract dishes from OCR text...")
                    dishes = self._extract_dishes_with_ai(text)
                    print(f"   ✅ Extracted {len(dishes)} dishes from image")
                    return dishes
                else:
                    print("   ⚠️  No AI client available for OCR extraction")
                    return []
            
            except ImportError:
                print("   ⚠️  pytesseract not installed. Install with: pip install pytesseract")
                print("   Also install Tesseract-OCR: https://github.com/tesseract-ocr/tesseract")
                return []
        
        except Exception as e:
            print(f"   ❌ Error scraping image menu: {e}")
            return []
    
    def _extract_dishes_with_ai(self, text: str) -> List[str]:
        """Use Groq AI to extract dish names from text"""
        if not self.groq_client:
            return []
        
        try:
            prompt = f"""Extract ONLY the actual dish names from this menu text. 
Do not include:
- Category names (appetizers, entrees, desserts, etc.)
- Cuisine types (Italian, Chinese, Thai, etc.)
- Descriptions or prices
- Generic food categories (pizza, pasta, burgers, etc.)

Return ONLY a comma-separated list of specific dish names.

Menu text:
{text}

Dish names (comma-separated):"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse comma-separated dishes
            dishes = [d.strip() for d in content.split(',')]
            dishes = [d for d in dishes if self._is_valid_dish_name(d)]
            
            return dishes[:30]
        
        except Exception as e:
            print(f"   ❌ AI extraction error: {e}")
            return []
    
    def _is_valid_dish_name(self, dish: str) -> bool:
        """Validate if text is a dish name (not category or junk)"""
        if not dish or len(dish) < 3:
            return False
        
        # Remove prices
        dish_clean = re.sub(r'\$?\d+(\.\d{2})?', '', dish).strip()
        if len(dish_clean) < 3:
            return False
        
        # Check if too long (likely description)
        if len(dish_clean) > 100:
            return False
        
        # Check if it's a category word
        dish_lower = dish_clean.lower()
        if dish_lower in self.category_words:
            return False
        
        # Must contain at least some letters
        if not re.search(r'[a-zA-Z]{2,}', dish_clean):
            return False
        
        return True


def main():
    """Example usage"""
    import json
    
    scraper = MenuURLScraper()
    
    # Test with a sample menu URL
    print("\n" + "="*60)
    print("MENU URL SCRAPER - TEST")
    print("="*60)
    
    test_urls = [
        "https://www.example-restaurant.com/menu",  # Replace with actual menu URL
    ]
    
    for url in test_urls:
        print(f"\n📋 Testing: {url}")
        dishes = scraper.scrape_menu_url(url)
        
        print(f"\n✅ Found {len(dishes)} dishes:")
        for i, dish in enumerate(dishes[:10], 1):
            print(f"   {i}. {dish}")
        
        if len(dishes) > 10:
            print(f"   ... and {len(dishes) - 10} more")


if __name__ == "__main__":
    main()
