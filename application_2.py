from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
from collections import Counter
import time # Import time for logging
import traceback # Import for detailed error logging
import os
from functools import wraps

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- Flask Setup ----------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
application = Flask(__name__)
# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------- Load Data and Models ---------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
print("Loading data and models for the recommender system...")
manga_image_links_df = pd.read_csv('title-link-image-score-eng-title.csv')
manga_data_raw = pd.read_csv('manga_list_3.csv')
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
interaction_sparse = joblib.load('interaction_sparse_centered.pkl')
svd = joblib.load('svd_model_centered.pkl')
nn_model = joblib.load('nn_model_centered.pkl')
user_means = joblib.load('user_means.pkl')
user_id_map = joblib.load('user_id_map.pkl')
title_id_map = joblib.load('title_id_map.pkl')
title_name_map = {name: i for i, name in title_id_map.items()}
print("All files loaded successfully.")

# ---------------------------------------------------------------------------------------------------------------------#
# ----------------------------------- Helper Functions ----------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
def recommend_manga_by_genre(genres, top_n=5):
    if not genres or not isinstance(genres, list): return pd.DataFrame()
    initial_filter = manga_data[manga_data['Genres'].notna() & manga_data['Genres'].str.contains('|'.join(genres), case=False, na=False)].copy()
    if initial_filter.empty: return pd.DataFrame()
    initial_filter['match_count'] = initial_filter['Genres'].apply(lambda manga_genres: sum(g.lower() in manga_genres.lower() for g in genres))
    top_manga = initial_filter.sort_values(by=['match_count', 'Score'], ascending=[False, False]).head(top_n)
    return top_manga[['Title', 'Score', 'Genres', 'Link', 'Thumbnail']]

def recommend_books_for_new_user(new_user_ratings, nn_model, svd, interaction_sparse, top_n=5):
    print("--- LOG: Using V1-style recommendation algorithm (raw user input, weighted average). ---")
    num_items = interaction_sparse.shape[1]
    new_user_vector = np.zeros(num_items)
    rated_title_ids = []
    for title, rating in new_user_ratings.items():
        if title in title_name_map:
            title_id = title_name_map[title]
            new_user_vector[title_id] = rating
            rated_title_ids.append(title_id)
    if not rated_title_ids: return []
    new_user_latent_vector = svd.transform(new_user_vector.reshape(1, -1))
    distances, indices = nn_model.kneighbors(new_user_latent_vector, n_neighbors=20)
    similar_user_indices = indices.flatten()
    similarity_scores = 1 - distances.flatten()
    similar_users_ratings = interaction_sparse[similar_user_indices]
    weighted_sum = similarity_scores.dot(similar_users_ratings.toarray())
    similarity_sum = np.sum(similarity_scores)
    if similarity_sum > 0: predicted_scores = weighted_sum / similarity_sum
    else: predicted_scores = np.zeros(num_items)
    predicted_scores[rated_title_ids] = -np.inf
    top_indices = np.argsort(predicted_scores)[-top_n:][::-1]
    recommended_titles = [title_id_map.get(i) for i in top_indices if title_id_map.get(i) is not None]
    print(f"--- LOG: Found recommendations (V1 logic): {recommended_titles} ---")
    return recommended_titles

# ---------------------------------------------------------------------------------------------------------------------#
# --------------------------------------- WEBSITE ROUTES --------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
@application.route('/')
def home(): return render_template("home.html")
@application.route('/resume')
def resume(): return render_template("resume.html")
@application.route('/projects')
def projects(): return render_template("projects.html")
@application.route('/project_specific')
def project_specific(): return render_template("project_specific.html")
@application.route('/manga_recommendation', methods=['GET', 'POST'])
def mangaRecommendation():
    manga_list = []
    genres = sorted(set(g.strip() for sublist in manga_data['Genres'].dropna().str.split(',') for g in sublist))
    if request.method == 'POST':
        selected_genres = [g for g in [request.form.get(f'genre_{i}') for i in range(1, 4)] if g]
        if selected_genres:
            recommendations = recommend_manga_by_genre(selected_genres, top_n=5)
            for _, row in recommendations.iterrows(): manga_list.append({"title": row['Title'], "url": row['Link'], "score": row['Score'], "genres": row['Genres'], "thumbnail": row['Thumbnail']})
        else:
            book_ratings = {}
            for i in range(1, 6):
                manga_title = request.form.get(f'manga_{i}')
                rating = request.form.get(f'rating_{i}')
                if manga_title and rating:
                    try: book_ratings[manga_title.strip()] = int(rating)
                    except ValueError: return "Invalid input. Please enter a number for ratings.", 400
            if book_ratings:
                try:
                    recommended_titles = recommend_books_for_new_user(book_ratings, nn_model, svd, interaction_sparse, top_n=5)
                    for title in recommended_titles:
                        manga_row = manga_data[manga_data['Title'] == title]
                        if not manga_row.empty:
                            row = manga_row.iloc[0]
                            manga_list.append({"title": row['Title'], "url": row['Link'], "thumbnail": row['Thumbnail'], "score": row['Score']})
                        else: manga_list.append({"title": title, "url": "#", "thumbnail": "https://via.placeholder.com/150x225", "score": "N/A"})
                except Exception as e:
                    print(f"Error during recommendation: {e}")
                    traceback.print_exc()
    return render_template("manga_recommendation.html", genres=genres, recommendations=manga_list)

# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------- API ROUTES -------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Server is up and running!"}), 200


@application.route('/api/recommend/user', methods=['POST'])
def api_recommend_user():
    try:
        book_ratings = request.get_json()
        if not book_ratings: return jsonify({"error": "Invalid input. Please provide ratings in JSON format."}), 400
        
        recommended_titles = recommend_books_for_new_user(book_ratings, nn_model, svd, interaction_sparse, top_n=10)
        
        # --- FIX: Filter the main DataFrame and handle NaN values before creating the response ---
        if not recommended_titles:
            return jsonify([])

        results_df = manga_data[manga_data['Title'].isin(recommended_titles)]
        
        # Convert NaN to None, which becomes 'null' in JSON (the correct way)
        results_df = results_df.where(pd.notnull(results_df), None)
        
        # Select relevant columns and convert to a list of dictionaries
        results = results_df[['title', 'url', 'thumbnail', 'score', 'genres']].to_dict(orient='records')

        return jsonify(results)
    except Exception as e:
        print(f"!!! SERVER ERROR in /api/recommend/user: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred while generating recommendations."}), 500


@application.route('/api/recommend/genre', methods=['POST'])
def api_recommend_genre():
    try:
        data = request.get_json()
        if not data or 'genres' not in data: return jsonify({"error": "Invalid input. Please provide a 'genres' list."}), 400
        
        selected_genres = data['genres']
        recommendations_df = recommend_manga_by_genre(selected_genres, top_n=10)
        
        # --- FIX: Handle potential NaN values before sending the response ---
        # Convert NaN to None, which becomes 'null' in JSON
        recommendations_df = recommendations_df.where(pd.notnull(recommendations_df), None)

        results = recommendations_df.to_dict(orient='records')
        return jsonify(results)
    except Exception as e:
        print(f"!!! SERVER ERROR in /api/recommend/genre: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

# Run the app
if __name__ == '__main__':
    application.run(host="localhost", port=5000, debug=True)
