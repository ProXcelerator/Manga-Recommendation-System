from flask import Flask, request, render_template, jsonify
from flask_cors import CORS # Import CORS
import pandas as pd
import numpy as np
import pickle
import traceback
from scipy.sparse import csr_matrix
from collections import Counter
from jikanpy import Jikan
jikan = Jikan()

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- Flask Setup ----------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
application = Flask(__name__)
# VITAL: Enable CORS for all routes, allowing your React Native app to connect
CORS(application) 

# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------- Load Data and Models ---------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
print("Loading data and models for the recommender system...")

# Safely load models with error handling
try:
    # Load the CSV file into a pandas DataFrame
    synopsis_df = pd.read_csv('synop.csv')
    
    # Create a 'normalized' title column for reliable matching (lowercase, no extra spaces)
    synopsis_df['normalized_title'] = synopsis_df['Title'].str.lower().str.strip()
    
    # Create the fast lookup dictionary: {normalized_title: Synopsis}
    # This is much faster than searching the DataFrame every time.
    synopsis_lookup = pd.Series(synopsis_df.Synopsis.values, index=synopsis_df.normalized_title).to_dict()
    
    print("synop.csv loaded successfully into lookup dictionary.")

except FileNotFoundError:
    print("Warning: synop.csv not found. Synopses will not be available.")
    synopsis_lookup = {} # Use an empty dict to prevent errors



try:
    # --- Load SVD Model and Mappings from Pickle Files ---
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    with open('title_map.pkl', 'rb') as f:
        title_map = pickle.load(f)
    with open('title_name_to_id.pkl', 'rb') as f:
        title_name_to_id = pickle.load(f)
    
    # --- MODIFIED SECTION: Load the Pre-processed Manga DataFrame ---
    # This is more efficient than reloading and processing the CSVs every time.
    with open('manga_metadata.pkl', 'rb') as f:
        manga_data = pickle.load(f)
    print("All model and data files loaded successfully.")

# --- Data Preprocessing (run once at startup) ---
    # MODIFIED: Updated to keep Genres and Themes separate.
    
    # --- Data Preprocessing (run once at startup) ---
    print("Starting data preprocessing...")
    # Fill any missing 'Genres' or 'Themes' with an empty string to prevent errors.
    manga_data['Genres'] = manga_data['Genres'].fillna('')
    manga_data['Themes'] = manga_data['Themes'].fillna('')
    
    # Create the 'Genre_Set' from ONLY the 'Genres' column.
    manga_data['Genre_Set'] = manga_data['Genres'].str.lower().str.split(',').apply(
        lambda tag_list: {tag.strip() for tag in tag_list if tag.strip()}
    )
    
    # Create the 'Thumbnail' column from the 'Image URL'.
    manga_data['Thumbnail'] = manga_data['Image URL'].fillna("https://placehold.co/150x225/2d3748/ffffff?text=No+Image")
    
    # Ensure no duplicate titles exist in the final DataFrame.
    manga_data.drop_duplicates(subset=['Title'], keep='first', inplace=True)
    
    # Create a sorted list for the UI dropdown menu from ONLY the genres.
    all_genres = sorted(list(set(g for genre_set in manga_data['Genre_Set'] for g in genre_set if g)))
    
    # --- NEW: Create a master lookup for Original and English titles ---
    any_title_to_original_title = {}
    for index, row in manga_data.iterrows():
        original_title = row['Title']
        english_title = row['English Title']
        
        # Map the lowercase original title to the proper-cased original title
        if pd.notna(original_title):
            any_title_to_original_title[original_title.lower()] = original_title
        
        # Map the lowercase English title to the proper-cased original title
        if pd.notna(english_title):
            any_title_to_original_title[english_title.lower()] = original_title
    
    print("✅ Data preprocessing complete.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: A required file was not found: {e}. The application cannot start.")
    manga_data = None
    svd_model = None
    all_genres = []

# ---------------------------------------------------------------------------------------------------------------------#
# ----------------------------------- Helper Functions ----------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#


import time # Add this to the top of your application_2.py file

def fetch_manga_from_jikan(title_query: str) -> dict | None:
    """
    Searches for a manga by title using the Jikan API, now with a delay
    to respect rate limits.
    """
    try:
        print(f"--- JIKAN LOG: Searching for '{title_query}' on Jikan API... ---")
        
        # ADDED: A 2-second delay to respect API rate limits.
        time.sleep(.5)
        
        search_results = jikan.search(search_type='manga', query=title_query, page=1)
        
        if not search_results.get('data'):
            print(f"--- JIKAN LOG: No results found for '{title_query}'. ---")
            return None
            
        manga = search_results['data'][0]
        
        genres_str = ', '.join([g.get('name') for g in manga.get('genres', []) if g.get('name')])
        themes_str = ', '.join([t.get('name') for t in manga.get('themes', []) if t.get('name')])

        formatted_manga = {
            'Title': manga.get('title'),
            'English Title': manga.get('title_english'),
            'URL': manga.get('url'),
            'Thumbnail': manga.get('images', {}).get('jpg', {}).get('image_url', "https://placehold.co/150x225/2d3748/ffffff?text=No+Image"),
            'Popularity': manga.get('popularity'),
            'Synopsis': manga.get('synopsis'),
            'Score': manga.get('score'),
            'Genres': genres_str,
            'Themes': themes_str
        }
        print(f"--- JIKAN LOG: Successfully found and formatted '{title_query}'. ---")
        return formatted_manga

    except Exception as e:
        print(f"--- JIKAN ERROR: An exception occurred while searching for '{title_query}': {e} ---")
        return None

def recommend_manga_by_genre(genres, top_n=10, excluded_titles=[]):
    """Recommends manga by finding the best genre matches, excluding certain titles."""
    if not genres or not isinstance(genres, list) or manga_data is None:
        return pd.DataFrame()
    
    target_genres = set(g.lower() for g in genres)
    
    temp_manga_data = manga_data.copy()
    
    temp_manga_data['match_count'] = temp_manga_data['Genre_Set'].apply(
        lambda manga_genres: len(target_genres.intersection(manga_genres))
    )
    
    excluded_lower = [t.lower() for t in excluded_titles]
    if excluded_titles:
        temp_manga_data = temp_manga_data[~temp_manga_data['Title'].str.lower().isin(excluded_lower)]

    top_manga = temp_manga_data[temp_manga_data['match_count'] > 0]
    top_manga = top_manga.sort_values(by=['match_count', 'Score'], ascending=[False, False]).head(top_n)

    # MODIFIED: Added 'English Title' to the list of columns being returned.
    return top_manga[['Title', 'English Title', 'Score', 'Genres', 'URL', 'Thumbnail']]


def get_hybrid_recommendations(user_ratings, n_recommendations=5):
    """
    Generates recommendations, matching user input against both original and English titles.
    """
    valid_ratings_for_svd = []
    input_titles = list(user_ratings.keys())

    for title, score in user_ratings.items():
        user_title_lower = title.lower()
        
        # Use the master lookup to find the canonical original title
        canonical_title = any_title_to_original_title.get(user_title_lower)
        
        # Check if the found title is in our SVD model's vocabulary
        if canonical_title and canonical_title in title_name_to_id:
            valid_ratings_for_svd.append((canonical_title, score))
        else:
            print(f"Info: '{title}' not in the collaborative filtering dataset. Will use for genre fallback.")
    
    final_recommendations = []
    excluded_titles = set(t.lower() for t in input_titles)

    # --- SVD Mode (Primary) ---
    if valid_ratings_for_svd:
        print("\n--- LOG: Using SVD collaborative filtering. ---")
        num_items = len(title_map)
        new_user_vector = np.zeros(num_items)
        rated_indices = []

        for title, score in valid_ratings_for_svd:
            title_id = title_name_to_id[title]
            new_user_vector[title_id] = score
            rated_indices.append(title_id)
            
        new_user_sparse_row = csr_matrix(new_user_vector.reshape(1, -1))
        new_user_latent_vector = svd_model.transform(new_user_sparse_row)
        predicted_scores = np.dot(new_user_latent_vector, svd_model.components_).flatten()
        predicted_scores[rated_indices] = -np.inf

        top_indices = np.argsort(predicted_scores)[- (n_recommendations * 3) :][::-1] 
        svd_recs = [title_map.get(i) for i in top_indices if title_map.get(i) is not None]
        
        for title in svd_recs:
            if len(final_recommendations) < n_recommendations and title.lower() not in excluded_titles:
                final_recommendations.append(title)
                excluded_titles.add(title.lower())
    
    # --- Fallback Loop to ensure 5 recommendations ---
    if len(final_recommendations) < n_recommendations:
        print(f"\n--- LOG: SVD gave {len(final_recommendations)} recs. Filling remaining slots. ---")
        
        titles_for_genre_search = input_titles + final_recommendations
        user_manga_genres_df = manga_data[
            manga_data['Title'].str.lower().isin([t.lower() for t in titles_for_genre_search]) |
            manga_data['English Title'].str.lower().isin([t.lower() for t in titles_for_genre_search])
        ]
        
        if not user_manga_genres_df.empty:
            genres_list = user_manga_genres_df['Genres'].dropna().str.split(',').tolist()
            all_genres_from_input = [genre.strip().lower() for sublist in genres_list for genre in sublist]
            top_two_genres = [genre for genre, count in Counter(all_genres_from_input).most_common(2)]
            
            if top_two_genres:
                print(f"--- LOG: Using top genres for fallback: {top_two_genres} ---")
                genre_recs_df = recommend_manga_by_genre(top_two_genres, top_n=10, excluded_titles=list(excluded_titles))
                for title in genre_recs_df['Title']:
                    if len(final_recommendations) < n_recommendations and title.lower() not in excluded_titles:
                        final_recommendations.append(title)
                        excluded_titles.add(title.lower())
        
        # 2. Top-Rated Fallback
        if len(final_recommendations) < n_recommendations:
            print(f"--- LOG: Still need more recs. Using top-rated overall. ---")
            top_rated = manga_data.sort_values(by='Score', ascending=False)
            
            for title in top_rated['Title']:
                if len(final_recommendations) < n_recommendations and title.lower() not in excluded_titles:
                    final_recommendations.append(title)
                    excluded_titles.add(title.lower())

    return final_recommendations[:n_recommendations]

# ---------------------------------------------------------------------------------------------------------------------#
# --------------------------------------- WEBSITE & API ROUTES --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/')
def home():
    """Serves the main home page."""
    return render_template("home.html")

@application.route('/projects')
def projects():
    """Serves the projects page."""
    return render_template("projects.html")

@application.route('/resume')
def resume():
    """Serves the resume page."""
    return render_template("resume.html")

@application.route('/manga_recommendation', methods=['GET', 'POST'])
def manga_recommendation_page():
    """Serves the original recommendation page (with genre and rating inputs)."""
    if manga_data is None:
        return "Error: Manga data is not loaded.", 500
        
    manga_list = []
    
    if request.method == 'POST':
        try:
            # Check for genre submissions first
            selected_genres = [g for g in [request.form.get(f'genre_{i}') for i in range(1, 4)] if g]
            
            # Then check for manga ratings
            book_ratings = {
                request.form.get(f'manga_{i}').strip(): int(request.form.get(f'rating_{i}'))
                for i in range(1, 6) if request.form.get(f'manga_{i}') and request.form.get(f'rating_{i}')
            }
            
            recommended_titles = []
            if book_ratings:
                recommended_titles = get_hybrid_recommendations(book_ratings, n_recommendations=5)
            elif selected_genres:
                recs_df = recommend_manga_by_genre(selected_genres, top_n=5)
                recommended_titles = recs_df['Title'].tolist()

            if recommended_titles:
                recs_df = manga_data[manga_data['Title'].isin(recommended_titles)].copy()
                title_cat = pd.CategoricalDtype(categories=recommended_titles, ordered=True)
                recs_df['Title'] = recs_df['Title'].astype(title_cat)
                recs_df = recs_df.sort_values('Title')
                manga_list = recs_df.to_dict(orient='records')

        except Exception as e:
            print(f"Error during recommendation: {e}")
            traceback.print_exc()

    return render_template("manga_recommendation.html", recommendations=manga_list, genres=all_genres)

@application.route('/manga_recommendation_ui', methods=['GET'])
def manga_recommendation_ui_page():
    """Serves the new UI page. Recommendation logic is now handled by an API call."""
    if manga_data is None:
        return "Error: Manga data is not loaded.", 500
    
    return render_template("manga_recommendation_UI.html", genres=all_genres)


@application.route('/genre_search')
def genre_search():
    """Serves the genre recommendation page and provides the genre list."""
    # This now passes the list of all genres to your HTML template
    return render_template("genre_search.html", genres=all_genres)

# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------- API ROUTES (For UI and other apps) -------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

# In the "API ROUTES" section of application_2.py

@application.route('/api/recommend', methods=['POST'])
def api_recommend():
    """
    API endpoint to get recommendations, with Jikan fallback for missing data.
    """
    if manga_data is None:
        return jsonify({"error": "Manga data not loaded on server."}), 500
        
    try:
        data = request.get_json()
        selected_genres = data.get('genres', [])
        book_ratings = data.get('ratings', {})

        recommended_titles = []
        if book_ratings:
            recommended_titles = get_hybrid_recommendations(book_ratings, n_recommendations=5)
        elif selected_genres:
            recs_df = recommend_manga_by_genre(selected_genres, top_n=5)
            recommended_titles = recs_df['Title'].tolist()

        if not recommended_titles:
            return jsonify([])

        final_recommendations = []
        for title in recommended_titles:
            local_result = manga_data[manga_data['Title'].str.lower() == title.lower()]
            
            if not local_result.empty:
                manga_details = local_result.iloc[0].to_dict()

                original_title = manga_details.get('Title', '')
                normalized_title = original_title.lower().strip()
                manga_details['Synopsis'] = synopsis_lookup.get(normalized_title, None)

                # --- UPDATED JIKAN FALLBACK LOGIC ---
                # Check if crucial data (synopsis, English title, or popularity) is missing
                is_data_missing = (
                    pd.isna(manga_details.get('English Title')) or not manga_details.get('English Title') or
                    not manga_details.get('Synopsis') or
                    pd.isna(manga_details.get('Popularity')) # <-- ADDED THIS CHECK
                )

                if is_data_missing:
                    print(f"--- LOG: Missing local data for '{title}'. Fetching from Jikan... ---")
                    jikan_result = fetch_manga_from_jikan(title)
                    if jikan_result:
                        manga_details.update(jikan_result)
                
                final_recommendations.append(manga_details)
            else:
                jikan_result = fetch_manga_from_jikan(title)
                if jikan_result:
                    final_recommendations.append(jikan_result)
        
        df = pd.DataFrame(final_recommendations)
        if 'Genre_Set' in df.columns:
             df['Genre_Set'] = df['Genre_Set'].apply(lambda x: list(x) if isinstance(x, set) else x)
        cleaned_recs = df.replace({np.nan: None}).to_dict(orient='records')

        return jsonify(cleaned_recs)

    except Exception as e:
        print(f"--- FATAL ERROR in /api/recommend: {e} ---")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during recommendation."}), 500

import numpy as np # Make sure numpy is imported at the top of your file

@application.route('/api/search', methods=['GET'])
def search_manga():
    """
    Provides autocomplete search results.
    - Searches both original and English titles.
    - Displays the English title in results if available, otherwise the original.
    """
    try:
        if manga_data is None:
            return jsonify({"error": "Manga data not loaded on server."}), 500
        query = request.args.get('q', '').lower()
        if len(query) < 3:
            return jsonify([])
        
        # Create boolean masks for both title columns
        mask_title = manga_data['Title'].str.lower().str.contains(query, na=False)
        mask_english = manga_data['English Title'].str.lower().str.contains(query, na=False)
        
        # Filter the DataFrame to get all rows that match in either column
        results = manga_data[mask_title | mask_english]
        
        # De-duplicate the results based on the unique original 'Title'
        unique_results = results.drop_duplicates(subset=['Title']).head(25)
        
        # Create a "Display Title" that prioritizes the English title
        # np.where(condition, value_if_true, value_if_false)
        display_titles = np.where(
            pd.notna(unique_results['English Title']), # Condition: If English Title exists...
            unique_results['English Title'],          # ...use it.
            unique_results['Title']                   # ...otherwise, use the original Title.
        )
        
        return jsonify(display_titles.tolist())

    except Exception as e:
        print(f"!!! SERVER ERROR in /api/search: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

# Run the app
if __name__ == '__main__':
    application.run(host="localhost", port=5000, debug=True)