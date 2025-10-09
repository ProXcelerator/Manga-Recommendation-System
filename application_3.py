from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import time
import traceback
from scipy.sparse import csr_matrix
from jikanpy import Jikan

# -----------------------------------------------------------------------------#
# --------------------------- SETUP & INITIALIZATION --------------------------#
# -----------------------------------------------------------------------------#

application = Flask(__name__)
jikan = Jikan()

print("🚀 Loading data and models for the recommender system...")
try:
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    with open('title_map.pkl', 'rb') as f:
        title_map = pickle.load(f)
    with open('title_name_to_id.pkl', 'rb') as f:
        title_name_to_id = pickle.load(f)
    with open('manga_metadata.pkl', 'rb') as f:
        manga_data = pickle.load(f)

    # --- Preprocessing for Genre Dropdowns ---
    manga_data['Genres'] = manga_data['Genres'].fillna('')
    genre_sets = manga_data['Genres'].str.lower().str.split(',').apply(
        lambda tag_list: {tag.strip() for tag in tag_list if tag.strip()}
    )
    all_genres = sorted(list(set(g for genre_set in genre_sets for g in genre_set if g)))
    
    print("✅ All model and data files loaded successfully.")

except FileNotFoundError as e:
    print(f"🔥 FATAL ERROR: A required file was not found: {e}. The application cannot start.")
    manga_data = None
    all_genres = []

# -----------------------------------------------------------------------------#
# ------------------------------ HELPER FUNCTIONS -----------------------------#
# -----------------------------------------------------------------------------#

def fetch_thumbnail_from_jikan(title_query: str) -> str | None:
    """
    Searches Jikan for a title and returns only the best image URL.
    """
    try:
        print(f"--- JIKAN LOG: Searching for thumbnail for '{title_query}'... ---")
        time.sleep(1) # Respect API rate limits
        search_results = jikan.search(search_type='manga', query=title_query, page=1)
        if search_results and search_results.get('data'):
            image_url = search_results['data'][0].get('images', {}).get('jpg', {}).get('image_url')
            if image_url:
                return image_url
        return None
    except Exception as e:
        print(f"--- JIKAN ERROR: Thumbnail search failed for '{title_query}': {e} ---")
        return None

from collections import Counter # Make sure this is imported at the top of your file

def get_recommendations(user_ratings: dict, n_recommendations: int = 5) -> list:
    """
    Generates 5 recommendations using SVD and a multi-stage fallback system.
    """
    # --- Step 1: Try the SVD Model ---
    valid_ratings_for_svd = []
    input_titles = list(user_ratings.keys())
    for title, score in user_ratings.items():
        found_title = next((key for key in title_name_to_id if key.lower() == title.lower()), None)
        if found_title:
            valid_ratings_for_svd.append((found_title, score))

    final_recommendations = []
    excluded_titles = {t[0] for t in valid_ratings_for_svd}

    if valid_ratings_for_svd:
        print("\n--- LOG: Using SVD collaborative filtering. ---")
        num_items = len(title_map)
        new_user_vector = np.zeros(num_items)
        for title, score in valid_ratings_for_svd:
            new_user_vector[title_name_to_id[title]] = score
        
        new_user_sparse_row = csr_matrix(new_user_vector)
        new_user_latent_vector = svd_model.transform(new_user_sparse_row)
        predicted_scores = np.dot(new_user_latent_vector, svd_model.components_).flatten()
        predicted_scores[[title_name_to_id[t[0]] for t in valid_ratings_for_svd]] = -np.inf

        top_indices = np.argsort(predicted_scores)[- (n_recommendations * 3):][::-1]
        
        for index in top_indices:
            title = title_map.get(index)
            if title and title not in excluded_titles:
                final_recommendations.append(title)
                excluded_titles.add(title)
                if len(final_recommendations) >= n_recommendations:
                    break

    # --- Step 2: Genre-Based Fallback ---
    if len(final_recommendations) < n_recommendations:
        print(f"--- LOG: SVD gave {len(final_recommendations)}. Using genre fallback. ---")
        user_manga_df = manga_data[manga_data['Title'].isin(input_titles)]
        if not user_manga_df.empty:
            genres_list = user_manga_df['Genres'].dropna().str.split(',').tolist()
            all_genres = [g.strip().lower() for sublist in genres_list for g in sublist]
            top_two_genres = [g for g, count in Counter(all_genres).most_common(2)]

            if top_two_genres:
                genre_recs_df = recommend_manga_by_genre(top_two_genres, top_n=10, excluded_titles=excluded_titles)
                for title in genre_recs_df['Title']:
                    if title not in excluded_titles:
                        final_recommendations.append(title)
                        excluded_titles.add(title)
                        if len(final_recommendations) >= n_recommendations:
                            break
    
    # --- Step 3: Top-Rated Fallback ---
    if len(final_recommendations) < n_recommendations:
        print(f"--- LOG: Still < 5 recs. Using top-rated fallback. ---")
        top_rated = manga_data.sort_values(by='Score', ascending=False)
        for title in top_rated['Title']:
            if title not in excluded_titles:
                final_recommendations.append(title)
                excluded_titles.add(title)
                if len(final_recommendations) >= n_recommendations:
                    break
    
    # --- Step 4: Enrich Final Titles with Full Details ---
    final_recs_details = []
    for title in final_recommendations[:n_recommendations]:
        local_record = manga_data[manga_data['Title'] == title]
        if not local_record.empty:
            manga_details = local_record.iloc[0].to_dict()
            new_thumbnail = fetch_thumbnail_from_jikan(title)
            manga_details['Thumbnail'] = new_thumbnail or manga_details.get('Image URL')
            final_recs_details.append(manga_details)
            
    return final_recs_details

def recommend_manga_by_genre(genres, top_n=10, excluded_titles=[]):
    """Recommends manga by finding the best genre matches."""
    if not genres or not isinstance(genres, list) or manga_data is None:
        return pd.DataFrame()
    
    target_genres = set(g.lower() for g in genres)
    
    # Pre-calculate genre sets if they don't exist
    if 'Genre_Set' not in manga_data.columns:
        manga_data['Genre_Set'] = manga_data['Genres'].str.lower().str.split(',').apply(
            lambda tag_list: {tag.strip() for tag in tag_list if isinstance(tag_list, list) and tag.strip()}
        )

    temp_manga_data = manga_data.copy()
    temp_manga_data['match_count'] = temp_manga_data['Genre_Set'].apply(
        lambda manga_genres: len(target_genres.intersection(manga_genres))
    )
    
    if excluded_titles:
        temp_manga_data = temp_manga_data[~temp_manga_data['Title'].isin(excluded_titles)]

    top_manga = temp_manga_data[temp_manga_data['match_count'] > 0]
    return top_manga.sort_values(by=['match_count', 'Score'], ascending=[False, False]).head(top_n)
# -----------------------------------------------------------------------------#
# ------------------------------ WEBSITE & API ROUTES -------------------------#
# -----------------------------------------------------------------------------#

@application.route('/manga_recommendation_ui')
def manga_recommendation_ui_page():
    """Serves the main HTML page."""
    return render_template("manga_recommendation_UI.html", genres=all_genres)

@application.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint that receives user ratings and returns full recommendation details."""
    if manga_data is None:
        return jsonify({"error": "Manga data not loaded on server."}), 500
    try:
        data = request.get_json()
        book_ratings = data.get('ratings', {})
        if not book_ratings:
            return jsonify([])
        recommendations = get_recommendations(book_ratings, n_recommendations=5)
        cleaned_recs = pd.DataFrame(recommendations).replace({np.nan: None}).to_dict(orient='records')
        return jsonify(cleaned_recs)
    except Exception as e:
        print(f"--- FATAL ERROR in /api/recommend: {e} ---")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during recommendation."}), 500

# -----------------------------------------------------------------------------#
# --------------------------------- RUN THE APP -------------------------------#
# -----------------------------------------------------------------------------#

if __name__ == '__main__':
    application.run(host="localhost", port=5000, debug=True)