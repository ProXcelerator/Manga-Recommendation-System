from flask import Flask, request, render_template, jsonify
from flask_cors import CORS # Import CORS
import pandas as pd
import numpy as np
import pickle
import traceback
from scipy.sparse import csr_matrix
from collections import Counter

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
    # --- Load SVD Model and Mappings ---
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    with open('title_map.pkl', 'rb') as f:
        title_map = pickle.load(f)
    with open('title_name_to_id.pkl', 'rb') as f:
        title_name_to_id = pickle.load(f)
    
    # --- Load General Manga Info for Display ---
    manga_data_raw = pd.read_csv('manga_list_3.csv')
    manga_image_links_df = pd.read_csv('title-link-image-score-eng-title.csv')
    print("All model and data files loaded successfully.")

    # --- Data Preprocessing (run once at startup) ---
    manga_data_raw['Title'] = manga_data_raw['Link'].apply(lambda x: x.split('/')[-1].replace('_', ' '))
    image_df_subset = manga_image_links_df[['Title', 'Image Link', 'Score_y']].copy()
    image_df_subset.rename(columns={'Image Link': 'Thumbnail', 'Score_y': 'Display Score'}, inplace=True)
    manga_data = pd.merge(manga_data_raw, image_df_subset, on='Title', how='left')
    manga_data['Thumbnail'] = manga_data['Thumbnail'].str.replace('/r/50x70', '', regex=False).fillna("https://placehold.co/150x225/2d3748/ffffff?text=No+Image")
    manga_data['Score'] = manga_data['Display Score'].fillna(manga_data['Score'])
    manga_data.drop_duplicates(subset=['Title'], keep='first', inplace=True)
    manga_data['Genre_Set'] = manga_data['Genres'].str.lower().str.split(',').apply(
        lambda g_list: set(g.strip() for g in g_list) if isinstance(g_list, list) else set()
    )
    print("Data preprocessing complete.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: A required file was not found: {e}. The application cannot start.")
    manga_data = None
    svd_model = None

# ---------------------------------------------------------------------------------------------------------------------#
# ----------------------------------- Helper Functions ----------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

def recommend_manga_by_genre(genres, top_n=10, excluded_titles=[]):
    """Recommends manga by finding the best genre matches, excluding certain titles."""
    if not genres or not isinstance(genres, list) or manga_data is None:
        return pd.DataFrame()
    
    target_genres = set(g.lower() for g in genres)
    
    # Create a copy to avoid SettingWithCopyWarning
    temp_manga_data = manga_data.copy()
    
    temp_manga_data['match_count'] = temp_manga_data['Genre_Set'].apply(
        lambda manga_genres: len(target_genres.intersection(manga_genres))
    )
    
    # Filter out manga that the user has already rated
    if excluded_titles:
        temp_manga_data = temp_manga_data[~temp_manga_data['Title'].str.lower().isin([t.lower() for t in excluded_titles])]

    top_manga = temp_manga_data[temp_manga_data['match_count'] > 0]
    top_manga = top_manga.sort_values(by=['match_count', 'Score'], ascending=[False, False]).head(top_n)

    return top_manga[['Title', 'Score', 'Genres', 'Link', 'Thumbnail']]


def get_hybrid_recommendations(user_ratings, n_recommendations=5):
    """
    Generates recommendations using SVD if possible. If a title is not found in the SVD model's
    dataset, it falls back to a genre-based recommendation system.
    """
    valid_ratings_for_svd = []
    input_titles = list(user_ratings.keys())

    for title, score in user_ratings.items():
        # Find case-insensitive match for the title
        found_title = next((key for key in title_name_to_id if key.lower() == title.lower()), None)
        if found_title:
            valid_ratings_for_svd.append((found_title, score))
        else:
            print(f"Info: '{title}' not in the collaborative filtering dataset. Will use for genre fallback.")

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
        
        # Ensure previously rated items are not recommended
        predicted_scores[rated_indices] = -np.inf

        top_indices = np.argsort(predicted_scores)[-n_recommendations:][::-1]
        return [title_map.get(i) for i in top_indices if title_map.get(i) is not None]

    # --- Genre-Based Fallback Mode ---
    else:
        print("\n--- LOG: No valid SVD titles found. Switching to genre-based fallback. ---")
        
        # 1. Find the genres for the user's input titles from the main manga_data
        user_manga_genres_df = manga_data[manga_data['Title'].str.lower().isin([t.lower() for t in input_titles])]
        
        if user_manga_genres_df.empty:
             # Ultimate fallback: if we don't even know the genre, return top rated overall
            print("--- LOG: Could not find genres for input. Returning top-rated manga. ---")
            top_rated = manga_data.sort_values(by='Score', ascending=False)
            return top_rated.head(n_recommendations)['Title'].tolist()

        # 2. Extract, flatten, and count all genres from the user's rated manga
        genres_list = user_manga_genres_df['Genres'].dropna().str.split(',').tolist()
        all_genres = [genre.strip().lower() for sublist in genres_list for genre in sublist]
        
        # 3. Find the two most common genres
        top_two_genres = [genre for genre, count in Counter(all_genres).most_common(2)]
        print(f"--- LOG: Using top genres for fallback: {top_two_genres} ---")

        # 4. Get recommendations based on these genres, excluding the titles the user already entered
        recommendations_df = recommend_manga_by_genre(top_two_genres, top_n=n_recommendations, excluded_titles=input_titles)
        return recommendations_df['Title'].tolist()

# ---------------------------------------------------------------------------------------------------------------------#
# --------------------------------------- WEBSITE & API ROUTES --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
@application.route('/')
def home(): 
    return render_template("home.html") # Assuming you have other pages

@application.route('/manga_recommendation', methods=['GET', 'POST'])
def manga_recommendation_page():
    if manga_data is None:
        return "Error: Manga data is not loaded.", 500
        
    manga_list = []
    
    if request.method == 'POST':
        try:
            # Collect all manga titles and ratings from the form
            book_ratings = {
                request.form.get(f'manga_{i}').strip(): int(request.form.get(f'rating_{i}'))
                for i in range(1, 4) if request.form.get(f'manga_{i}') and request.form.get(f'rating_{i}')
            }
            
            if book_ratings:
                recommended_titles = get_hybrid_recommendations(book_ratings, n_recommendations=5)
                recs_df = manga_data[manga_data['Title'].isin(recommended_titles)]
                
                # Reorder results to match the recommendation order, which can be important
                if recommended_titles:
                    recs_df = recs_df.set_index('Title').loc[recommended_titles].reset_index()

                manga_list = recs_df.to_dict(orient='records')
        except Exception as e:
            print(f"Error during recommendation: {e}")
            traceback.print_exc()

    return render_template("manga_recommendation.html", recommendations=manga_list)

# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------- API ROUTES (For React Native, etc.) ------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/api/search', methods=['GET'])
def search_manga():
    """Provides autocomplete search results for manga titles."""
    try:
        if manga_data is None:
            return jsonify({"error": "Manga data not loaded on server."}), 500
        query = request.args.get('q', '').lower()
        if len(query) < 3:
            return jsonify([]) # Don't search for very short strings
        
        # Search for titles containing the query
        results = manga_data[manga_data['Title'].str.lower().str.contains(query, na=False)].head(10)
        results_list = results['Title'].tolist()
        return jsonify(results_list)
    except Exception as e:
        print(f"!!! SERVER ERROR in /api/search: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

# (Keep other API routes if they are used by your React Native app)

# Run the app
if __name__ == '__main__':
    application.run(host="localhost", port=5000, debug=True)
