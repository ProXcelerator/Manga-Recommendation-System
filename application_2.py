from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from collections import Counter
import time # Import time for logging

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- Flask Setup ----------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
application = Flask(__name__)

# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------- Load Data and Models (UPDATED) -----------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
print("Loading data and models for the recommender system...")

# Load the dataset of manga titles and links for displaying results
manga_image_links_df = pd.read_csv('title-link-image-score-eng-title.csv')

# Load dataset containing manga and genres for the genre-based recommendations
manga_data_raw = pd.read_csv('manga_list_3.csv')
manga_data_raw['Title'] = manga_data_raw['Link'].apply(lambda x: x.split('/')[-1].replace('_', ' '))

# --- FIX: Merge data sources into a single, reliable dataframe ---
image_df_subset = manga_image_links_df[['Title', 'Image Link', 'Score_y']].copy()
image_df_subset.rename(columns={'Image Link': 'Thumbnail', 'Score_y': 'Display Score'}, inplace=True)
manga_data = pd.merge(manga_data_raw, image_df_subset, on='Title', how='left')
manga_data['Thumbnail'] = manga_data['Thumbnail'].str.replace('/r/50x70', '', regex=False).fillna("https://via.placeholder.com/150x225")
manga_data['Score'] = manga_data['Display Score'].fillna(manga_data['Score'])
manga_data.drop_duplicates(subset=['Title'], keep='first', inplace=True)

# --- OPTIMIZATION: Pre-process genres for faster filtering ---
manga_data['Genre_Set'] = manga_data['Genres'].str.lower().str.split(',').apply(
    lambda g_list: set(g.strip() for g in g_list) if isinstance(g_list, list) else set()
)

# Load the NEW mean-centered models and data files
interaction_sparse = joblib.load('interaction_sparse_centered.pkl')
svd = joblib.load('svd_model_centered.pkl')
nn_model = joblib.load('nn_model_centered.pkl') 
user_means = joblib.load('user_means.pkl')
user_id_map = joblib.load('user_id_map.pkl')
title_id_map = joblib.load('title_id_map.pkl')

# Create a reverse map for easy lookup of title names to their integer IDs
title_name_map = {name: i for i, name in title_id_map.items()}

print("All files loaded successfully.")

# ---------------------------------------------------------------------------------------------------------------------#
# ----------------------------------- Helper Functions (UPDATED) ------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

def recommend_manga_by_genre(genres, top_n=5):
    if not genres or not isinstance(genres, list):
        return pd.DataFrame() 

    initial_filter = manga_data[
        manga_data['Genres'].notna() & manga_data['Genres'].str.contains('|'.join(genres), case=False, na=False)
    ].copy()

    if initial_filter.empty:
        return pd.DataFrame()

    initial_filter['match_count'] = initial_filter['Genres'].apply(
        lambda manga_genres: sum(g.lower() in manga_genres.lower() for g in genres)
    )

    top_manga = initial_filter.sort_values(
        by=['match_count', 'Score'], ascending=[False, False]
    ).head(top_n)
    
    return top_manga[['Title', 'Score', 'Genres', 'Link', 'Thumbnail']]

def recommend_books_for_new_user(new_user_ratings, nn_model, svd, interaction_sparse, top_n=5):
    start_time = time.time()
    print(f"--- LOG: Starting recommendation process at {start_time:.2f} ---")

    # --- Tier 1: Attempt Hybrid KNN + Genre Filtering ---
    new_user_avg_rating = np.mean(list(new_user_ratings.values()))
    num_items = interaction_sparse.shape[1]
    new_user_vector = np.zeros(num_items)
    rated_title_ids, input_manga_titles = [], []

    for title, rating in new_user_ratings.items():
        if title in title_name_map:
            title_id = title_name_map[title]
            new_user_vector[title_id] = rating - new_user_avg_rating
            rated_title_ids.append(title_id)
            input_manga_titles.append(title)
    
    new_user_reduced = svd.transform(new_user_vector.reshape(1, -1))
    distances, indices = nn_model.kneighbors(new_user_reduced, n_neighbors=20)
    
    candidate_title_ids = set()
    similar_user_ratings = interaction_sparse[indices.flatten()]
    for i in range(similar_user_ratings.shape[0]):
        user_row = similar_user_ratings.getrow(i)
        positive_mask = user_row.data > 0
        candidate_title_ids.update(user_row.indices[positive_mask])
    
    candidate_title_ids.difference_update(rated_title_ids)
    
    if candidate_title_ids:
        all_user_genres_list = []
        input_manga_genres_df = manga_data[manga_data['Title'].isin(input_manga_titles)]['Genres'].dropna()
        for genre_list in input_manga_genres_df.str.split(','):
            all_user_genres_list.extend([g.strip().lower() for g in genre_list])

        if all_user_genres_list:
            genre_counts = Counter(all_user_genres_list)
            required_genres = {genre for genre, count in genre_counts.items() if count > 1}
            
            candidate_df = manga_data[manga_data['Title'].isin([title_id_map.get(i) for i in candidate_title_ids])].copy()
            final_candidates = pd.DataFrame()

            if required_genres:
                mask = candidate_df['Genre_Set'].apply(required_genres.issubset)
                final_candidates = candidate_df[mask]

            if final_candidates.empty:
                user_genres_set = set(all_user_genres_list)
                if user_genres_set:
                    mask = candidate_df['Genre_Set'].apply(lambda x: not x.isdisjoint(user_genres_set))
                    final_candidates = candidate_df[mask]

            # If genre filtering produced results, return them
            if not final_candidates.empty:
                print("--- LOG: Success with Primary Hybrid Method. ---")
                top_recommendations = final_candidates.sort_values(by='Score', ascending=False).head(top_n)
                return top_recommendations['Title'].tolist()

    # --- Tier 2: Fallback to KNN without Genre Filter ---
    print("--- LOG: Primary method failed. Falling back to KNN without genre filter. ---")
    if candidate_title_ids:
        candidate_df = manga_data[manga_data['Title'].isin([title_id_map.get(i) for i in candidate_title_ids])].copy()
        top_recommendations = candidate_df.sort_values(by='Score', ascending=False).head(top_n)
        return top_recommendations['Title'].tolist()

    # --- Tier 3: Fallback to Content-Based (Most Common Genre) ---
    print("--- LOG: KNN failed. Falling back to most common genre. ---")
    if 'all_user_genres_list' in locals() and all_user_genres_list:
        most_common_genre = Counter(all_user_genres_list).most_common(1)[0][0]
        print(f"--- LOG: Most common genre is '{most_common_genre}'. ---")
        genre_recommendations = manga_data[
            manga_data['Genre_Set'].apply(lambda x: most_common_genre in x) &
            (~manga_data['Title'].isin(input_manga_titles))
        ].sort_values(by='Score', ascending=False).head(top_n)
        return genre_recommendations['Title'].tolist()

    # --- Tier 4: Ultimate Fallback to Global Top Rated ---
    print("--- LOG: All methods failed. Falling back to global top rated. ---")
    top_rated = manga_data[~manga_data['Title'].isin(input_manga_titles)].sort_values(by='Score', ascending=False).head(top_n)
    return top_rated['Title'].tolist()


# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------- Flask Routes ----------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
@application.route('/')
def home():
    return render_template("home.html")

@application.route('/resume')
def resume():
    return render_template("resume.html")

@application.route('/projects')
def projects():
    return render_template("projects.html")

@application.route('/project_specific')
def project_specific():
    return render_template("project_specific.html")

@application.route('/manga_recommendation', methods=['GET', 'POST'])
def mangaRecommendation():
    manga_list = []
    genres = sorted(set(
        g.strip() for sublist in manga_data['Genres'].dropna().str.split(',') for g in sublist
    ))

    if request.method == 'POST':
        selected_genres = [g for g in [request.form.get(f'genre_{i}') for i in range(1, 4)] if g]

        if selected_genres:
            recommendations = recommend_manga_by_genre(selected_genres, top_n=5)
            for _, row in recommendations.iterrows():
                manga_list.append({
                    "title": row['Title'], "url": row['Link'], "score": row['Score'],
                    "genres": row['Genres'], "thumbnail": row['Thumbnail']
                })
        else:
            book_ratings = {}
            for i in range(1, 6):
                manga_title = request.form.get(f'manga_{i}')
                rating = request.form.get(f'rating_{i}')
                if manga_title and rating:
                    try:
                        book_ratings[manga_title.strip()] = int(rating)
                    except ValueError:
                        return "Invalid input. Please enter a number for ratings.", 400

            if book_ratings:
                print(f"--- LOG: Received book ratings: {book_ratings} ---")
                recommended_titles = recommend_books_for_new_user(book_ratings, nn_model, svd, interaction_sparse, top_n=5)
                print(f"--- LOG: Function returned: {recommended_titles} ---")
                for title in recommended_titles:
                    manga_row = manga_data[manga_data['Title'] == title]
                    if not manga_row.empty:
                        row = manga_row.iloc[0]
                        manga_list.append({
                            "title": row['Title'], "url": row['Link'],
                            "thumbnail": row['Thumbnail'], "score": row['Score']
                        })
                    else:
                        manga_list.append({
                            "title": title, "url": "#", "thumbnail": "https://via.placeholder.com/150x225", "score": "N/A"
                        })

    return render_template("manga_recommendation.html", genres=genres, recommendations=manga_list)

# Run the app
if __name__ == '__main__':
    application.run(host="localhost", port=5000, debug=True)

