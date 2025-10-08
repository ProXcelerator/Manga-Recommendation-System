from flask import Flask, request, render_template, jsonify
from flask_cors import CORS # Import CORS
import pandas as pd
import numpy as np
import pickle
import traceback
from scipy.sparse import csr_matrix

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
    with open('manga_metadata.pkl', 'rb') as f:
        manga_metadata = pickle.load(f)
    
    # --- Load General Manga Info for Display ---
    manga_data_raw = pd.read_csv('manga_list_3.csv')
    manga_image_links_df = pd.read_csv('title-link-image-score-eng-title.csv')
    print("All model and data files loaded successfully.")

    # --- Data Preprocessing (run once at startup) ---
    manga_data_raw['Title'] = manga_data_raw['Link'].apply(lambda x: x.split('/')[-1].replace('_', ' '))
    image_df_subset = manga_image_links_df[['Title', 'Image Link', 'Score_y']].copy()
    image_df_subset.rename(columns={'Image Link': 'Thumbnail', 'Score_y': 'Display Score'}, inplace=True)
    manga_data = pd.merge(manga_data_raw, image_df_subset, on='Title', how='left')
    manga_data['Thumbnail'] = manga_data['Thumbnail'].str.replace('/r/50x70', '', regex=False).fillna("https://via.placeholder.com/150x225")
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

def recommend_manga_by_genre(genres, top_n=5):
    """Recommends manga by finding the best genre matches."""
    if not genres or not isinstance(genres, list) or manga_data is None:
        return pd.DataFrame()
    
    target_genres = set(g.lower() for g in genres)
    manga_data['match_count'] = manga_data['Genre_Set'].apply(
        lambda manga_genres: len(target_genres.intersection(manga_genres))
    )
    top_manga = manga_data[manga_data['match_count'] > 0].copy()
    top_manga = top_manga.sort_values(by=['match_count', 'Score'], ascending=[False, False]).head(top_n)

    return top_manga[['Title', 'Score', 'Genres', 'Link', 'Thumbnail']]


def hybrid_recommend_for_new_user(user_ratings, n_recommendations=5):
    """
    Generates recommendations using SVD if possible, otherwise falls back to a top-rated list.
    """
    valid_ratings_for_svd = []
    input_titles = list(user_ratings.keys())

    for title, score in user_ratings.items():
        found_title = next((key for key in title_name_to_id if key.lower() == title.lower()), None)
        if found_title:
            valid_ratings_for_svd.append((found_title, score))
        else:
            print(f"Info: '{title}' not in the collaborative filtering dataset.")

    # --- SVD Mode ---
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

        top_indices = np.argsort(predicted_scores)[-n_recommendations:][::-1]
        return [title_map.get(i) for i in top_indices if title_map.get(i) is not None]

    # --- Fallback Mode ---
    else:
        print("\n--- LOG: No valid titles found. Switching to fallback mode. ---")
        if manga_metadata is None:
            return ["Could not provide fallback recommendations because metadata is missing."]
        
        top_rated = manga_metadata.sort_values(by='Score_y', ascending=False)
        recommendations = top_rated[~top_rated['Title'].str.lower().isin([t.lower() for t in input_titles])]
        return recommendations.head(n_recommendations)['Title'].tolist()

# ---------------------------------------------------------------------------------------------------------------------#
# --------------------------------------- WEBSITE & API ROUTES --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
@application.route('/')
def home(): 
    return render_template("home.html")

@application.route('/resume')
def resume(): return render_template("resume.html")

@application.route('/projects')
def projects(): return render_template("projects.html")

@application.route('/project_specific')
def project_specific(): return render_template("project_specific.html")

@application.route('/manga_recommendation', methods=['GET', 'POST'])
def manga_recommendation_page():
    if manga_data is None:
        return "Error: Manga data is not loaded.", 500
        
    all_genres = sorted(list(manga_data['Genre_Set'].explode().dropna().unique()))
    manga_list = []
    
    if request.method == 'POST':
        try:
            selected_genres = [g for g in [request.form.get(f'genre_{i}') for i in range(1, 4)] if g]
            if selected_genres:
                recommendations = recommend_manga_by_genre(selected_genres, top_n=5)
                manga_list = recommendations.to_dict(orient='records')
            else:
                book_ratings = {
                    request.form.get(f'manga_{i}').strip(): int(request.form.get(f'rating_{i}'))
                    for i in range(1, 6) if request.form.get(f'manga_{i}') and request.form.get(f'rating_{i}')
                }
                
                if book_ratings:
                    recommended_titles = hybrid_recommend_for_new_user(book_ratings, n_recommendations=5)
                    recs_df = manga_data[manga_data['Title'].isin(recommended_titles)]
                    # Reorder results to match recommendation order
                    recs_df = recs_df.set_index('Title').loc[recommended_titles].reset_index()
                    manga_list = recs_df.to_dict(orient='records')
        except Exception as e:
            print(f"Error during recommendation: {e}")
            traceback.print_exc()

    return render_template("manga_recommendation.html", genres=all_genres, recommendations=manga_list)

# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------- API ROUTES (Now with Safety Nets) --------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Server is up and running!"}), 200

@application.route('/api/genres', methods=['GET'])
def get_all_genres():
    try:
        if manga_data is None:
            return jsonify({"error": "Manga data not loaded on server."}), 500
        all_genres = sorted(list(manga_data['Genre_Set'].explode().dropna().unique()))
        return jsonify(all_genres)
    except Exception as e:
        print(f"!!! SERVER ERROR in /api/genres: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred while fetching genres."}), 500

@application.route('/api/search', methods=['GET'])
def search_manga():
    try:
        if manga_data is None:
            return jsonify({"error": "Manga data not loaded on server."}), 500
        query = request.args.get('q', '').lower()
        if len(query) < 3:
            return jsonify([])
        
        results = manga_data[manga_data['Title'].str.lower().str.contains(query, na=False)].head(20)
        results_list = results[['Title', 'Thumbnail']].to_dict(orient='records')
        return jsonify(results_list)
    except Exception as e:
        print(f"!!! SERVER ERROR in /api/search: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

@application.route('/api/top-manga', methods=['GET'])
def top_manga():
    try:
        if manga_data is None:
            return jsonify({"error": "Manga data not loaded on server."}), 500
        # Ensure 'Score' column is numeric before sorting
        manga_data['Score'] = pd.to_numeric(manga_data['Score'], errors='coerce')
        top_20 = manga_data.sort_values('Score', ascending=False).head(20)
        results_list = top_20[['Title', 'Thumbnail']].to_dict(orient='records')
        return jsonify(results_list)
    except Exception as e:
        print(f"!!! SERVER ERROR in /api/top-manga: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

@application.route('/api/recommend/user', methods=['POST'])
def api_recommend_user():
    if not svd_model:
        return jsonify({"error": "Recommendation model is not loaded"}), 500
    try:
        book_ratings = request.get_json()
        if not book_ratings:
            return jsonify({"error": "Invalid input. Please provide ratings in JSON format."}), 400
        
        recommended_titles = hybrid_recommend_for_new_user(book_ratings, n_recommendations=10)
        
        results_df = manga_data[manga_data['Title'].isin(recommended_titles)]
        results = []
        for title in recommended_titles:
            manga_row = results_df[results_df['Title'] == title]
            if not manga_row.empty:
                row = manga_row.iloc[0]
                results.append({
                    "title": row['Title'],
                    "url": row['Link'],
                    "thumbnail": row['Thumbnail'],
                    "score": row['Score'] if pd.notna(row['Score']) else None,
                    "genres": row['Genres'] if pd.notna(row['Genres']) else None
                })
        return jsonify(results)
    except Exception as e:
        print(f"!!! SERVER ERROR in /api/recommend/user: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

@application.route('/api/recommend/genre', methods=['POST'])
def api_recommend_genre():
    try:
        if manga_data is None:
            return jsonify({"error": "Manga data not loaded on server."}), 500
        data = request.get_json()
        if not data or 'genres' not in data:
            return jsonify({"error": "Invalid input. Please provide a 'genres' list."}), 400
        
        selected_genres = data['genres']
        recommendations_df = recommend_manga_by_genre(selected_genres, top_n=10)
        
        # Clean NaN values for safe JSON conversion
        recommendations_df['Score'] = recommendations_df['Score'].where(pd.notna(recommendations_df['Score']), None)
        recommendations_df['Genres'] = recommendations_df['Genres'].where(pd.notna(recommendations_df['Genres']), None)

        results = recommendations_df.to_dict(orient='records')
        return jsonify(results)
    except Exception as e:
        print(f"!!! SERVER ERROR in /api/recommend/genre: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

# Run the app
if __name__ == '__main__':
    application.run(host="localhost", port=5000, debug=True)

