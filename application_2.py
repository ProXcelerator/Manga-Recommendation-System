from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import traceback
from jikanpy import Jikan
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# --- NEW: SQLAlchemy and PostgreSQL Imports ---
import os
from sqlalchemy import create_engine, text, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
# --- END NEW ---

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- Database Setup -------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

# --- MODIFIED: Prioritize the internal URL from the server environment ---
# On your Render server, this will automatically use the fast, secure internal URL.
# The hardcoded string is kept as a fallback for local development.
DATABASE_URL = os.environ.get(
    "postgresql://mangadb_z6qn_user:D5HZWxfDzqIaP9UDRWHUCiCBr7ZwCXZB@dpg-d3kjah95pdvs739jfro0-a/mangadb_z6qn", 
    "postgresql://mangadb_z6qn_user:D5HZWxfDzqIaP9UDRWHUCiCBr7ZwCXZB@dpg-d3kjah95pdvs739jfro0-a.ohio-postgres.render.com/mangadb_z6qn"
)
# --- END MODIFIED ---


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Define the Manga table schema to match the one in seed_database.py
class Manga(Base):
    __tablename__ = "manga"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, unique=True, index=True)
    english_title = Column(String, nullable=True)
    synopsis = Column(String, nullable=True)
    genres = Column(String, nullable=True)
    themes = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    thumbnail = Column(String, nullable=True)
    url = Column(String, nullable=True)
    embedding = Column(Vector(384))

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- Flask & Model Setup --------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

application = Flask(__name__)
CORS(application) 

jikan = Jikan()

print("--- Loading ML Models into Memory... ---")
# These models are still needed for on-the-fly processing
st_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=st_model) # Efficiently reuse the model
print("--- ✅ ML Models Loaded. ---")

# ---------------------------------------------------------------------------------------------------------------------#
# ----------------------------------- Helper Functions (Now DB-driven) ------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

import time

def fetch_manga_from_jikan_and_save(title_query: str, db_session) -> Manga | None:
    """
    Searches Jikan for a manga, generates its embedding, and saves it to the database.
    """
    try:
        print(f"--- JIKAN LOG: Searching for '{title_query}'... ---")
        time.sleep(0.5)
        search_results = jikan.search(search_type='manga', query=title_query, page=1)
        
        if not search_results.get('data'): return None
            
        manga_api = search_results['data'][0]

        # Don't save if synopsis is missing
        if not manga_api.get('synopsis'): return None

        # Generate embedding for the new synopsis
        embedding = st_model.encode(manga_api['synopsis'])

        new_manga = Manga(
            title=manga_api.get('title'),
            english_title=manga_api.get('title_english'),
            synopsis=manga_api.get('synopsis'),
            genres=', '.join([g.get('name') for g in manga_api.get('genres', [])]),
            themes=', '.join([t.get('name') for t in manga_api.get('themes', [])]),
            score=manga_api.get('score'),
            thumbnail=manga_api.get('images', {}).get('jpg', {}).get('image_url'),
            url=manga_api.get('url'),
            embedding=embedding
        )
        
        db_session.add(new_manga)
        db_session.commit()
        print(f"--- DB LOG: Saved new manga '{new_manga.title}' from Jikan. ---")
        return new_manga

    except Exception as e:
        db_session.rollback()
        print(f"--- JIKAN/DB ERROR: {e} ---")
        return None

def recommend_manga_by_synopsis(query_title, selected_themes=[], top_k=5):
    """
    Finds similar manga using pgvector for efficiency.
    """
    db = SessionLocal()
    try:
        # Find the manga in our database (case-insensitive search)
        manga = db.query(Manga).filter(Manga.title.ilike(f'%{query_title}%')).first()

        # If not found, fetch from Jikan, save, and then use it
        if not manga:
            manga = fetch_manga_from_jikan_and_save(query_title, db)

        if not manga or manga.embedding is None:
            return []

        # --- Vector Similarity Search using pgvector ---
        # The `<=>` operator finds the cosine distance (0=identical, 2=opposite)
        # We find more candidates to re-rank them later.
        candidates = db.query(Manga).order_by(Manga.embedding.cosine_distance(manga.embedding)).limit(100).all()

        # Re-ranking and filtering logic remains similar
        final_recommendations = []
        query_keywords = set(kw[0] for kw in kw_model.extract_keywords(manga.synopsis, top_n=10))

        for candidate in candidates:
            # Skip self and variations
            if candidate.id == manga.id or query_title.lower() in candidate.title.lower():
                continue

            shared_keywords = []
            if candidate.synopsis:
                rec_keywords = set(kw[0] for kw in kw_model.extract_keywords(candidate.synopsis, top_n=10))
                shared_keywords = list(query_keywords.intersection(rec_keywords))

            # If themes selected, only include recommendations that match at least one
            if selected_themes:
                if not any(theme in rec_keywords for theme in selected_themes):
                    continue
            
            final_recommendations.append({"manga": candidate, "shared_keywords": shared_keywords})

            if len(final_recommendations) >= top_k:
                break
        
        return final_recommendations

    finally:
        db.close()

# ---------------------------------------------------------------------------------------------------------------------#
# --------------------------------------- WEBSITE & API ROUTES --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/')
def home(): return render_template("home.html")

@application.route('/synopsis_search')
def synopsis_search_page(): return render_template("synopsis_search.html")

# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------- API ROUTES (Now DB-driven) ---------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/api/get_keywords', methods=['POST'])
def api_get_keywords():
    db = SessionLocal()
    try:
        data = request.get_json()
        query_title = data.get('title')
        if not query_title: return jsonify({"error": "A 'title' must be provided."}), 400

        manga = db.query(Manga).filter(Manga.title.ilike(f'%{query_title}%')).first()
        synopsis = manga.synopsis if manga else None

        if not synopsis:
            # Temporary Jikan fetch without saving for keyword extraction
            jikan_manga = fetch_manga_from_jikan_and_save(query_title, db)
            if jikan_manga: synopsis = jikan_manga.synopsis
        
        if not synopsis: return jsonify([])

        keywords = [kw[0] for kw in kw_model.extract_keywords(synopsis, top_n=12)]
        return jsonify(keywords)
    finally:
        db.close()

@application.route('/api/synopsis_recommend', methods=['POST'])
def api_synopsis_recommend():
    try:
        data = request.get_json()
        query_title = data.get('title')
        selected_themes = data.get('themes', [])
        if not query_title: return jsonify({"error": "A 'title' must be provided."}), 400

        recommendations = recommend_manga_by_synopsis(query_title, selected_themes=selected_themes, top_k=5)

        if not recommendations: return jsonify([])

        # Format the data for the frontend
        cleaned_recs = []
        for rec in recommendations:
            m = rec['manga']
            cleaned_recs.append({
                "Title": m.title,
                "English Title": m.english_title,
                "Synopsis": m.synopsis,
                "Genres": m.genres,
                "Thumbnail": m.thumbnail,
                "URL": m.url,
                "Score": m.score,
                "shared_keywords": rec['shared_keywords']
            })
        
        return jsonify(cleaned_recs)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred."}), 500

@application.route('/api/search', methods=['GET'])
def search_manga():
    db = SessionLocal()
    try:
        query = request.args.get('q', '').lower()
        if len(query) < 3: return jsonify([])
        
        # Search both titles using ILIKE for case-insensitive matching
        results = db.query(Manga).filter(
            (Manga.title.ilike(f'%{query}%')) | (Manga.english_title.ilike(f'%{query}%'))
        ).limit(25).all()

        display_titles = [m.english_title if m.english_title else m.title for m in results]
        return jsonify(list(set(display_titles))) # Use set to ensure unique titles
    finally:
        db.close()

# Run the app
if __name__ == '__main__':
    application.run(host="localhost", port=5000, debug=True)

