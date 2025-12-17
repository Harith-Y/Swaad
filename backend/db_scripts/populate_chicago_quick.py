"""
Quick Chicago Population - Streamlined Version
Populates Chicago restaurants with error handling and progress saving
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from master_populate import RestaurantPopulator


def main():
    """Populate Chicago restaurants - streamlined"""
    
    print("\n" + "="*80)
    print("🏙️  CHICAGO RESTAURANT POPULATION (Quick Mode)")
    print("="*80)
    print("\nPopulating Chicago restaurants with dishes...")
    print("This will run unattended and save progress automatically.\n")
    
    # Just Chicago
    locations = ["Chicago, IL"]
    
    # Create populator
    populator = RestaurantPopulator()
    
    try:
        print(f"⏳ Starting population at {datetime.now().strftime('%H:%M:%S')}")
        print("   (This may take 20-30 minutes. Progress is saved automatically.)\n")
        
        populator.populate_locations(locations, limit_per_category=50)
        
        print(f"\n✅ Chicago population complete!")
        print(f"   Total restaurants: {len(populator.all_restaurants)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print(f"   Progress saved: {len(populator.all_restaurants)} restaurants collected")
        print("   Run again to continue from where you left off.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Progress saved: {len(populator.all_restaurants)} restaurants")
    
    # Always save what we have
    if populator.all_restaurants:
        output_file = f"chicago_restaurants_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(populator.all_restaurants, f, indent=2)
        print(f"\n💾 Saved to {output_file}")


if __name__ == "__main__":
    main()
