# Synopsis File Cleaner
# This script reads the synop.csv file, cleans up the 'Synopsis' column,
# and overwrites the file with the corrected data.
# Run this once to fix your existing file.
#
# Required libraries: pandas
# To install it, run this command in your terminal:
# pip install pandas

import pandas as pd
import os

# --- Configuration ---
CSV_FILENAME = 'synop.csv'

def clean_synopsis_text(synopsis):
    """Cleans a single synopsis string."""
    # Ensure the input is a string before processing
    if not isinstance(synopsis, str):
        return ''

    # Split the synopsis at the unwanted phrase and take everything before it.
    if '[Written by MAL Rewrite]' in synopsis:
        synopsis = synopsis.split('[Written by MAL Rewrite]')[0]
    
    # Final cleanup of any lingering whitespace
    return synopsis.strip()

def main():
    """Main function to run the cleaning tool."""
    print(f"--- Starting Cleanup for '{CSV_FILENAME}' ---")
    
    if not os.path.exists(CSV_FILENAME):
        print(f"Error: '{CSV_FILENAME}' not found. Nothing to clean.")
        return

    try:
        df = pd.read_csv(CSV_FILENAME)
        
        if 'Synopsis' not in df.columns:
            print(f"Error: 'Synopsis' column not found in '{CSV_FILENAME}'.")
            return
            
        print("Cleaning synopses...")
        # Apply the cleaning function to every row in the 'Synopsis' column
        df['Synopsis'] = df['Synopsis'].apply(clean_synopsis_text)
        
        # Save the cleaned data back to the original file
        df.to_csv(CSV_FILENAME, index=False)
        print(f"Cleanup complete. '{CSV_FILENAME}' has been updated.")

    except Exception as e:
        print(f"An error occurred during the cleaning process: {e}")

if __name__ == '__main__':
    main()
