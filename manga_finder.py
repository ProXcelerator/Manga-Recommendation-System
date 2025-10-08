# Manga Discovery Tool
# This script finds new manga titles from the Jikan API that are not already in your local CSV file.
#
# Required libraries: pandas, requests
# To install them, run this command in your terminal:
# pip install pandas requests

import pandas as pd
import requests
import time
import os

# --- Configuration ---
API_BASE_URL = 'https://api.jikan.moe/v4/manga'
CSV_FILENAME = 'title-link-image-score-eng-title.csv'
REQUEST_DELAY_SECONDS = 1  # Delay between API calls to respect rate limits.

def load_existing_manga(filename):
    """
    Loads the manga list from the specified CSV file.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing a pandas DataFrame of the existing manga
               and a set of existing manga titles for quick lookup.
               Returns an empty DataFrame and an empty set if the file doesn't exist.
    """
    if not os.path.exists(filename):
        print(f"Warning: '{filename}' not found. A new file will be created.")
        return pd.DataFrame(), set()

    try:
        df = pd.read_csv(filename)
        # Use the 'Title' column for the lookup set
        existing_titles = set(df['Title'].astype(str).unique())
        print(f"Successfully loaded {len(existing_titles)} unique titles from '{filename}'.")
        return df, existing_titles
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame(), set()

def fetch_manga_page(page_number):
    """
    Fetches a single page of manga data from the Jikan API.

    Args:
        page_number (int): The page number to fetch.

    Returns:
        dict: The JSON response from the API, or None if the request fails.
    """
    params = {'page': page_number, 'order_by': 'popularity', 'sort': 'asc'}
    try:
        print(f"Fetching page {page_number} from Jikan API...")
        response = requests.get(API_BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data from the API: {e}")
        return None

def parse_manga_data(manga_item):
    """
    Parses a single manga item from the API response into a dictionary
    that matches the CSV format.

    Args:
        manga_item (dict): A dictionary representing a single manga.

    Returns:
        dict: A dictionary with keys matching the CSV columns.
    """
    title = manga_item.get('title', '')
    english_title = manga_item.get('title_english', '')

    # Replicate the "Original Title" format from the CSV
    original_title = title
    if english_title and english_title != title:
        original_title = f"{title}\n{english_title}"

    # Join genres into a single string
    genres = ', '.join([genre['name'] for genre in manga_item.get('genres', [])])

    return {
        'Title': title,
        'Link': manga_item.get('url', ''),
        'Original Title': original_title,
        'English Title': english_title or '', # Ensure empty string instead of None
        'Genres': genres,
        'Score_y': manga_item.get('score'),
        'Image Link': manga_item.get('images', {}).get('webp', {}).get('small_image_url', '')
    }

def save_manga_list(df, filename):
    """
    Saves the combined manga list back to the CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The path to the CSV file.
    """
    try:
        # Save the DataFrame to CSV, overwriting the old file.
        # The first column `Unnamed: 0` will be correctly managed by pandas.
        df.to_csv(filename, index=False)
        print(f"\nSuccessfully saved updated manga list to '{filename}'.")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


def main():
    """Main function to run the manga discovery tool."""
    existing_df, existing_titles = load_existing_manga(CSV_FILENAME)
    new_manga_rows = []
    
    # Fetch the first page to get pagination info
    first_page_data = fetch_manga_page(1)
    total_pages = 0
    if first_page_data and 'pagination' in first_page_data:
        total_pages = first_page_data['pagination'].get('last_visible_page', 0)
        print(f"FYI: The API has a total of {total_pages} pages of manga.")
    else:
        print("Could not retrieve total page count from the API.")

    try:
        num_pages_to_scan = int(input("How many pages of popular manga do you want to scan? (e.g., 5): "))
    except ValueError:
        print("Invalid input. Please enter a number. Defaulting to 5 pages.")
        num_pages_to_scan = 5

    try:
        # Process the first page we already fetched
        if first_page_data and 'data' in first_page_data:
            for manga_item in first_page_data['data']:
                title = manga_item.get('title')
                if title and title not in existing_titles:
                    print(f"  -> Found new manga: '{title}'")
                    parsed_data = parse_manga_data(manga_item)
                    new_manga_rows.append(parsed_data)
                    existing_titles.add(title)
            time.sleep(REQUEST_DELAY_SECONDS)

        # Loop through the rest of the requested pages
        for page in range(2, num_pages_to_scan + 1):
            data = fetch_manga_page(page)

            if not data or 'data' not in data or not data['data']:
                print("No more data found on the API. Stopping.")
                break

            for manga_item in data['data']:
                title = manga_item.get('title')
                if title and title not in existing_titles:
                    print(f"  -> Found new manga: '{title}'")
                    parsed_data = parse_manga_data(manga_item)
                    new_manga_rows.append(parsed_data)
                    existing_titles.add(title) # Avoid adding duplicates found in the same session

            # Respectful delay between API calls
            time.sleep(REQUEST_DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving found manga...")

    if not new_manga_rows:
        print("\nScan complete. No new manga titles were found.")
        return

    print(f"\nFound a total of {len(new_manga_rows)} new manga titles.")
    new_df = pd.DataFrame(new_manga_rows)

    # Combine the old and new data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # The 'Unnamed: 0' column is often created when the index is saved as a column.
    # We will drop it and let pandas create a clean index on save.
    if 'Unnamed: 0' in combined_df.columns:
        combined_df = combined_df.drop(columns=['Unnamed: 0'])

    save_manga_list(combined_df, CSV_FILENAME)

if __name__ == '__main__':
    main()

