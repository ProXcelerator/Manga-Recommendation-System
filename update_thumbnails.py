import pandas as pd
import requests
import time
import os

# --- Configuration ---
INPUT_CSV = 'title-link-image-score-eng-title.csv'
JIKAN_API_URL = 'https://api.jikan.moe/v4/manga'
PLACEHOLDER_URL = 'https://via.placeholder.com/150x225'
# Rate limit for Jikan is 3 requests per second. Using 1 second is safe.
SLEEP_TIME = 1 
# --- End Configuration ---

def fetch_and_update_thumbnails():
    """
    1. Loads the CSV.
    2. Identifies missing/placeholder thumbnail links.
    3. Queries the Jikan API to find a new, valid link.
    4. Updates the DataFrame and saves it.
    """
    print(f"1. Loading data from {INPUT_CSV}...")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: CSV file not found at {INPUT_CSV}")
        return

    try:
        # Load the dataframe you use to store links
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Identify rows where 'Image Link' is the placeholder or is missing (NaN)
    missing_mask = (df['Image Link'].isna()) | (df['Image Link'] == PLACEHOLDER_URL)
    
    # Filter to only the rows we need to search for
    manga_to_search = df[missing_mask].copy()

    if manga_to_search.empty:
        print("✅ No missing thumbnail links found. The CSV is up-to-date.")
        return

    print(f"2. Found {len(manga_to_search)} manga titles with missing thumbnails.")

    # --- Start Search and Update Loop ---
    updated_count = 0
    
    for index, row in manga_to_search.iterrows():
        title = row['Title']
        print(f"\nSearching for: {title}")

        try:
            # Construct the API query
            params = {'q': title, 'limit': 1}
            response = requests.get(JIKAN_API_URL, params=params, timeout=10)
            response.raise_for_status() # Raises an exception for HTTP error codes

            data = response.json()
            
            # Check for results and extract the high-quality image URL
            if data and data.get('data'):
                first_result = data['data'][0]
                new_thumbnail_url = first_result['images']['jpg']['large_image_url']
                
                # Check if the title is a good match (simple case-insensitive comparison)
                api_title = first_result['title']
                
                # Simple check: if the title is in the API result's title
                if title.lower() in api_title.lower() or api_title.lower() in title.lower():
                    # Update the original DataFrame directly
                    original_index = df[df['Title'] == title].index[0]
                    df.loc[original_index, 'Image Link'] = new_thumbnail_url
                    print(f"   --> Found and updated with URL: {new_thumbnail_url}")
                    updated_count += 1
                else:
                    print(f"   --> Title mismatch: Found '{api_title}' instead of '{title}'. Skipping.")
            else:
                print(f"   --> No results found for '{title}'. Skipping.")

        except requests.exceptions.RequestException as e:
            print(f"   --> API Request Error for '{title}': {e}. Skipping.")
        except Exception as e:
            print(f"   --> General Error for '{title}': {e}. Skipping.")

        # Respect the API rate limit
        time.sleep(SLEEP_TIME)

    # --- Save Results ---
    if updated_count > 0:
        print(f"\n4. Successfully updated {updated_count} thumbnail links.")
        
        # Save the updated DataFrame back to the CSV, overwriting the old file
        df.to_csv(INPUT_CSV, index=False)
        print(f"✅ Changes saved to {INPUT_CSV}")
    else:
        print("\n4. No new links were successfully found and updated.")

if __name__ == '__main__':
    fetch_and_update_thumbnails()