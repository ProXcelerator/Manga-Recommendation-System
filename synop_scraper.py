# Scraper for synop.csv
# This script finds new manga from the Jikan API and adds their synopses to the synopsis CSV file.
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
CSV_FILENAME = 'synop.csv'
REQUEST_DELAY_SECONDS = 1

def load_manga_list(filename, title_column='Title'):
    """Loads the manga list from the CSV file."""
    if not os.path.exists(filename):
        print(f"Warning: '{filename}' not found. A new file will be created.")
        cols = ['Title', 'Synopsis']
        return pd.DataFrame(columns=cols), set()
    try:
        df = pd.read_csv(filename)
        if title_column not in df.columns:
            print(f"Error: Title column '{title_column}' not found in '{filename}'.")
            return pd.DataFrame(), set()
        existing_titles = set(df[title_column].dropna().astype(str).str.strip().unique())
        print(f"Successfully loaded {len(existing_titles)} unique titles from '{filename}'.")
        return df, existing_titles
    except Exception as e:
        print(f"Error reading CSV file '{filename}': {e}")
        return pd.DataFrame(), set()

def fetch_manga_page(page_number):
    """Fetches a single page of manga data from the Jikan API."""
    params = {'page': page_number, 'order_by': 'popularity', 'sort': 'asc'}
    try:
        print(f"Fetching page {page_number} from Jikan API...")
        response = requests.get(API_BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data from the API: {e}")
        return None

def parse_manga_data(item):
    """Parses API data into the format for the synopsis CSV file."""
    return {'Title': item.get('title', '').strip(), 'Synopsis': item.get('synopsis', '') if item.get('synopsis') else ''}

def save_manga_list(df, filename):
    """Saves the DataFrame back to the CSV file."""
    try:
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        df.to_csv(filename, index=False)
        print(f"Successfully saved updated list to '{filename}'.")
    except Exception as e:
        print(f"Error saving data to '{filename}': {e}")

def main():
    """Main function to run the scraping tool."""
    df, existing_titles = load_manga_list(CSV_FILENAME)
    new_rows = []
    
    try:
        num_pages_to_scan = int(input(f"How many pages to scan for '{CSV_FILENAME}'? (e.g., 5): "))
    except ValueError:
        print("Invalid input. Defaulting to 5 pages.")
        num_pages_to_scan = 5

    try:
        for page in range(1, num_pages_to_scan + 1):
            page_data = fetch_manga_page(page)
            if not page_data or 'data' not in page_data or not page_data['data']:
                print("No more data from API. Stopping scan.")
                break
            for item in page_data['data']:
                title = item.get('title', '').strip()
                if title and title not in existing_titles:
                    print(f"  -> Adding '{title}' to {CSV_FILENAME}")
                    new_rows.append(parse_manga_data(item))
                    existing_titles.add(title)
            time.sleep(REQUEST_DELAY_SECONDS)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Saving any manga found so far...")

    if new_rows:
        print(f"\nFound {len(new_rows)} new entries. Saving to '{CSV_FILENAME}'...")
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([df, new_df], ignore_index=True)
        save_manga_list(combined_df, CSV_FILENAME)
    else:
        print(f"\nScan complete. No new entries found for '{CSV_FILENAME}'.")

if __name__ == '__main__':
    main()
