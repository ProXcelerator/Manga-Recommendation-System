from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from collections import Counter

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
manga_data = pd.read_csv('manga_list_3.csv')
manga_data['Title'] = manga_data['Link'].apply(lambda x: x.split('/')[-1].replace('_', ' '))

# --- ADDED FEATURE: Prevent duplicate manga by cleaning data on load ---
manga_data.drop_duplicates(subset=['Title'], keep='first', inplace=True)
manga_image_links_df.drop_duplicates(subset=['Title'], keep='first', inplace=True)

# Load the NEW mean-centered models and data files
interaction_sparse = joblib.load('interaction_sparse_centered.pkl')
svd = joblib.load('svd_model_centered.pkl')
nn_model = joblib.load('nn_model_centered.pkl') # Re-loading the KNN model for this approach
user_means = joblib.load('user_means.pkl')
user_id_map = joblib.load('user_id_map.pkl')
title_id_map = joblib.load('title_id_map.pkl')

# Create a reverse map for easy lookup of title names to their integer IDs
title_name_map = {name: i for i, name in title_id_map.items()}

print("All files loaded successfully.")

# ---------------------------------------------------------------------------------------------------------------------#
# ----------------------------------- Helper Functions (UPDATED) ------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

# --- THIS FUNCTION IS UPDATED FOR MORE ACCURATE GENRE MATCHING ---
def recommend_manga_by_genre(genres, top_n=5):
    """
    Recommends manga by prioritizing titles that match more of the selected genres.
    """
    if not genres or not isinstance(genres, list):
        return pd.DataFrame() # Return empty DataFrame on error

    # First, efficiently filter the DataFrame to only include manga that contain at least one of the selected genres.
    # We use .copy() to avoid potential SettingWithCopyWarning later.
    initial_filter = manga_data[
        manga_data['Genres'].notna() & manga_data['Genres'].str.contains('|'.join(genres), case=False, na=False)
    ].copy()

    # If no manga match, return an empty frame.
    if initial_filter.empty:
        return pd.DataFrame()

    # Second, calculate how many of the selected genres each manga actually contains.
    # This creates a 'match_count' that we can use to rank relevance.
    initial_filter['match_count'] = initial_filter['Genres'].apply(
        lambda manga_genres: sum(g.lower() in manga_genres.lower() for g in genres)
    )

    # Finally, sort the results. Prioritize the highest match_count first (most relevant),
    # and then use the Score as a tie-breaker (most popular).
    top_manga = initial_filter.sort_values(
        by=['match_count', 'Score'], ascending=[False, False]
    ).head(top_n)
    
    return top_manga[['Title', 'Score', 'Genres', 'Link', 'Image Link']]

# --- THIS FUNCTION IS REWRITTEN FOR A HYBRID (KNN + CONTENT) APPROACH ---
def recommend_books_for_new_user(new_user_ratings, nn_model, svd, interaction_sparse, top_n=5):
    """
    Recommends manga using a hybrid KNN and content-based filtering approach:
    1. Finds similar users based on ratings (KNN Collaborative Filtering).
    2. Gathers manga highly rated by these similar users.
    3. Filters this list to match genres from the user's input (Content-based).
    4. Ranks the final list by score and returns the top N.
    """
    # 1. Prepare the new user's data
    new_user_avg_rating = np.mean(list(new_user_ratings.values()))
    num_items = interaction_sparse.shape[1]
    new_user_vector = np.zeros(num_items)
    rated_title_ids = []
    input_manga_titles = [] 

    for title, rating in new_user_ratings.items():
        if title in title_name_map:
            title_id = title_name_map[title]
            centered_score = rating - new_user_avg_rating
            new_user_vector[title_id] = centered_score
            rated_title_ids.append(title_id)
            input_manga_titles.append(title)

    # 2. Find similar users using KNN
    new_user_reduced = svd.transform(new_user_vector.reshape(1, -1))
    distances, indices = nn_model.kneighbors(new_user_reduced, n_neighbors=20) # Use more neighbors for a better pool
    similar_user_indices = indices.flatten()

    # 3. Aggregate manga from similar users
    # Get all items rated positively (score > 0 in centered matrix) by similar users
    similar_user_ratings = interaction_sparse[similar_user_indices]
    candidate_items = np.where(similar_user_ratings.toarray() > 0)
    candidate_title_ids = set(candidate_items[1]) # Get unique title IDs

    # Remove items the new user has already rated
    candidate_title_ids.difference_update(rated_title_ids)

    # If no candidates, return empty list
    if not candidate_title_ids:
        return []

    # 4. Filter candidates by genre, prioritizing a "main genre"
    # Get all genres from the user's input manga and count their occurrences
    all_user_genres_list = []
    input_manga_genres_df = manga_data[manga_data['Title'].isin(input_manga_titles)]['Genres'].dropna()
    for genre_list in input_manga_genres_df.str.split(','):
        all_user_genres_list.extend([g.strip().lower() for g in genre_list])

    if not all_user_genres_list:
        return [] # Cannot filter if no genres are found

    genre_counts = Counter(all_user_genres_list)
    # A "main genre" is one that appears more than once in the input
    required_genres = {genre for genre, count in genre_counts.items() if count > 1}
    
    candidate_df = manga_data[manga_data['Title'].isin([title_id_map.get(i) for i in candidate_title_ids])].copy()
    final_candidates = pd.DataFrame()

    if required_genres:
        # Strict filtering: manga must contain ALL required main genres
        def has_all_required_genres(manga_genre_str):
            if pd.isna(manga_genre_str): return False
            manga_genres_set = {g.strip().lower() for g in manga_genre_str.split(',')}
            return required_genres.issubset(manga_genres_set)
        
        genre_match_mask = candidate_df['Genres'].apply(has_all_required_genres)
        final_candidates = candidate_df[genre_match_mask]

    # If strict filtering yields no results, or if there were no main genres, fall back to lenient filtering
    if final_candidates.empty:
        user_genres_set = set(all_user_genres_list)
        if user_genres_set:
            genre_match_mask = candidate_df['Genres'].str.lower().str.contains('|'.join(user_genres_set), na=False)
            final_candidates = candidate_df[genre_match_mask]

    # 5. Rank by score and return the top N
    top_recommendations = final_candidates.sort_values(by='Score', ascending=False).head(top_n)

    return top_recommendations['Title'].tolist()

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
                image_link = row['Image Link'] if pd.notna(row['Image Link']) else "https://via.placeholder.com/150x225"
                high_res_image_link = image_link.replace('/r/50x70', '') if pd.notna(image_link) else image_link
                manga_list.append({
                    "title": row['Title'], "url": row['Link'], "score": row['Score'],
                    "genres": row['Genres'], "thumbnail": high_res_image_link
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
                recommended_titles = recommend_books_for_new_user(book_ratings, nn_model, svd, interaction_sparse, top_n=5)

                for title in recommended_titles:
                    manga_row = manga_image_links_df[
                        (manga_image_links_df['Title'].str.lower() == title.lower()) |
                        (manga_image_links_df['English Title'].str.lower() == title.lower())
                    ]
                    
                    link, thumbnail, score = "#", "https://via.placeholder.com/150x225", "N/A"
                    if not manga_row.empty:
                        row = manga_row.iloc[0]
                        link = row.get('Link', "#")
                        image_link = row.get('Image Link')
                        thumbnail = image_link.replace('/r/50x70', '') if pd.notna(image_link) else thumbnail
                        score = row.get('Score_y', "N/A")

                    manga_list.append({"title": title, "url": link, "thumbnail": thumbnail, "score": score})

    return render_template("manga_recommendation.html", genres=genres, recommendations=manga_list)

# Run the app
if __name__ == '__main__':
    application.run(host="localhost", port=5000, debug=True)

