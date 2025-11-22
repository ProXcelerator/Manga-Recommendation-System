from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
import pickle
import traceback
import psycopg2
import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from jikanpy import Jikan
from groq import Groq, RateLimitError, APIError
from flask_session import Session  
import secrets                     

# ---------------------------------------------------------------------------#
# ------------------------- CONFIGURATION & SETUP ---------------------------#
# ---------------------------------------------------------------------------#

application = Flask(__name__)
CORS(application)

# --- NEW SESSION CONFIGURATION ---
application.config["SECRET_KEY"] = secrets.token_hex(16) # Secure key for signing
application.config["SESSION_TYPE"] = "filesystem"        # Store history in files
application.config["SESSION_PERMANENT"] = False          # Clear when browser closes
Session(application)                                     # Initialize session

jikan = Jikan()

# Database Configuration
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "bakers",
    "host": "127.0.0.1",
    "port": "5431" 
}

# Initialize Groq Client
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

# ---------------------------------------------------------------------------#
# --------------------------- DATABASE FUNCTIONS ----------------------------#
# ---------------------------------------------------------------------------#

def get_db_connection():
    # 1. Check if we are on Render (getting the URL from the environment)
    db_url = os.environ.get("DATABASE_URL")
    
    if db_url:
        try:
            # Connect using the Render URL
            conn = psycopg2.connect(db_url)
            return conn
        except Exception as e:
            print(f"❌ Error connecting to Render DB: {e}")
            return None
    
    # 2. Fallback: If no URL is found, use Local Config (Development)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Error connecting to Local DB: {e}")
        return None

def save_manga_to_db(manga_data):
    conn = get_db_connection()
    if not conn: return

    try:
        cur = conn.cursor()
        norm_title = manga_data['Title'].lower().strip()
        cur.execute("SELECT id FROM manga_library WHERE normalized_title = %s", (norm_title,))
        if not cur.fetchone():
            print(f"💾 Saving new manga '{manga_data['Title']}' to DB...")
            insert_q = """
            INSERT INTO manga_library 
            (title, english_title, normalized_title, url, image_url, score, genres, themes, synopsis)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cur.execute(insert_q, (
                manga_data['Title'],
                manga_data.get('English Title'),
                norm_title,
                manga_data.get('URL'),
                manga_data.get('Thumbnail'),
                manga_data.get('Score'),
                manga_data.get('Genres'),
                manga_data.get('Themes'),
                manga_data.get('Synopsis')
            ))
            conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ DB Save Error: {e}")

def db_search_titles(query_str):
    conn = get_db_connection()
    if not conn: return []
    try:
        cur = conn.cursor()
        sql = """
        SELECT title, english_title FROM manga_library 
        WHERE title ILIKE %s OR english_title ILIKE %s 
        LIMIT 15
        """
        search_term = f"%{query_str}%"
        cur.execute(sql, (search_term, search_term))
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        final_list = []
        for r in results:
            if r[1]: final_list.append(r[1])
            else: final_list.append(r[0])
        return list(set(final_list))
    except Exception as e:
        print(f"Search Error: {e}")
        return []

def db_get_manga_details(title):
    conn = get_db_connection()
    if not conn: return None
    try:
        cur = conn.cursor()
        norm_title = title.strip().lower()
        
        sql = """
        SELECT title, english_title, url, image_url, score, genres, themes, synopsis 
        FROM manga_library WHERE normalized_title = %s LIMIT 1
        """
        cur.execute(sql, (norm_title,))
        row = cur.fetchone()
        
        if not row:
            sql_fuzzy = """
            SELECT title, english_title, url, image_url, score, genres, themes, synopsis 
            FROM manga_library WHERE title ILIKE %s OR english_title ILIKE %s LIMIT 1
            """
            cur.execute(sql_fuzzy, (title, title))
            row = cur.fetchone()
            
        cur.close()
        conn.close()
        
        if row:
            return {
                'Title': row[0],
                'English Title': row[1],
                'URL': row[2],
                'Thumbnail': row[3],
                'Score': row[4],
                'Genres': row[5],
                'Themes': row[6],
                'Synopsis': row[7]
            }
        return None
    except Exception as e:
        print(f"Details Error: {e}")
        return None

# --- UPDATED FUNCTION: SAFE SEARCH ---
def db_search_by_synopsis(user_query):
    """
    Finds manga by keyword but filters out NSFW genres unless explicitly requested.
    """
    conn = get_db_connection()
    if not conn: return []
    try:
        cur = conn.cursor()
        
        # 1. Detect if user is explicitly asking for adult content
        nsfw_triggers = ['hentai', 'erotica', 'porn', 'doujinshi', 'sex', '18+']
        is_nsfw_request = any(trigger in user_query.lower() for trigger in nsfw_triggers)

        # 2. Construct the Safety Filter
        # If it's NOT an NSFW request, we exclude those genres
        safety_clause = ""
        if not is_nsfw_request:
            safety_clause = """
                AND (
                    COALESCE(genres, '') NOT ILIKE '%Hentai%' 
                    AND COALESCE(genres, '') NOT ILIKE '%Erotica%'
                    AND COALESCE(themes, '') NOT ILIKE '%Adult Cast%'
                )
            """

        # 3. Run the Search
        sql = f"""
        SELECT title, english_title, url, image_url, score, genres, themes, synopsis,
               ts_rank(
                   to_tsvector('english', 
                       COALESCE(title, '') || ' ' || 
                       COALESCE(synopsis, '') || ' ' || 
                       COALESCE(genres, '') || ' ' || 
                       COALESCE(themes, '')
                   ), 
                   websearch_to_tsquery('english', %s)
               ) * (COALESCE(score, 5) / 10.0) as final_rank
        FROM manga_library
        WHERE to_tsvector('english', 
                   COALESCE(title, '') || ' ' || 
                   COALESCE(synopsis, '') || ' ' || 
                   COALESCE(genres, '') || ' ' || 
                   COALESCE(themes, '')
              ) @@ websearch_to_tsquery('english', %s)
        {safety_clause}
        ORDER BY final_rank DESC
        LIMIT 5;
        """
        
        cur.execute(sql, (user_query, user_query))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                'Title': row[0],
                'English Title': row[1],
                'URL': row[2],
                'Thumbnail': row[3],
                'Score': row[4],
                'Genres': row[5],
                'Themes': row[6],
                'Synopsis': row[7]
            })
        return results

    except Exception as e:
        print(f"Smart Search Error: {e}")
        return []

# ---------------------------------------------------------------------------#
# -------------------------- LOAD SVD MODELS --------------------------------#
# ---------------------------------------------------------------------------#
print("Loading SVD models...")
try:
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    with open('title_map.pkl', 'rb') as f:
        title_map = pickle.load(f)
    with open('title_name_to_id.pkl', 'rb') as f:
        title_name_to_id = pickle.load(f)
            
    print("✅ Models loaded. Running in Database Mode.")

except FileNotFoundError as e:
    print(f"🔥 FATAL ERROR: Missing model files. {e}")

# ---------------------------------------------------------------------------#
# -------------------------- LOGIC FUNCTIONS --------------------------------#
# ---------------------------------------------------------------------------#

def fetch_manga_from_jikan(title_query):
    try:
        time.sleep(0.5)
        search_results = jikan.search(search_type='manga', query=title_query, page=1)
        if not search_results.get('data'): return None
        manga = search_results['data'][0]
        
        data = {
            'Title': manga.get('title'),
            'English Title': manga.get('title_english'),
            'URL': manga.get('url'),
            'Thumbnail': manga.get('images', {}).get('jpg', {}).get('image_url'),
            'Popularity': manga.get('popularity'),
            'Synopsis': manga.get('synopsis'),
            'Score': manga.get('score'),
            'Genres': ', '.join([g['name'] for g in manga.get('genres', [])]),
            'Themes': ', '.join([t['name'] for t in manga.get('themes', [])])
        }
        save_manga_to_db(data)
        return data
    except Exception: return None

def get_hybrid_recommendations(user_ratings, n_recommendations=5):
    valid_ratings = []
    input_titles = list(user_ratings.keys())
    
    for title, score in user_ratings.items():
        if title in title_name_to_id:
            valid_ratings.append((title, score))
        else:
            for db_title in title_name_to_id.keys():
                if db_title.lower() == title.lower():
                     valid_ratings.append((db_title, score))
                     break
    
    final_recs = []
    excluded = set(t.lower() for t in input_titles)
    
    if valid_ratings:
        user_vec = np.zeros(len(title_map))
        rated_idx = []
        for t, s in valid_ratings:
            idx = title_name_to_id[t]
            user_vec[idx] = s
            rated_idx.append(idx)
            
        latent_vec = svd_model.transform(csr_matrix(user_vec.reshape(1, -1)))
        scores = np.dot(latent_vec, svd_model.components_).flatten()
        scores[rated_idx] = -np.inf 
        
        top_idx = np.argsort(scores)[-(n_recommendations*3):][::-1]
        for i in top_idx:
            rec_title = title_map.get(i)
            if rec_title and rec_title.lower() not in excluded:
                final_recs.append(rec_title)
                excluded.add(rec_title.lower())
                if len(final_recs) >= n_recommendations: break
    return final_recs[:n_recommendations]

def get_full_manga_data_smart(title):
    db_data = db_get_manga_details(title)
    if db_data: return db_data
    print(f"🌍 Fetching '{title}' from Jikan API...")
    return fetch_manga_from_jikan(title)

# ---------------------------------------------------------------------------#
# -------------------------- WEB ROUTES -------------------------------------#
# ---------------------------------------------------------------------------#

@application.route('/')
def home(): return render_template("home.html")

@application.route('/genre_search')
def genre_search(): return render_template("genre_search.html", genres=[])

@application.route('/chatbot')
def chatbot_page(): return render_template("chatbot.html")

@application.route('/resume')
def resume(): return render_template("resume.html")

@application.route('/projects')
def projects(): return render_template("projects.html")

# ---------------------------------------------------------------------------#
# -------------------------- API ROUTES -------------------------------------#
# ---------------------------------------------------------------------------#

@application.route('/api/search', methods=['GET'])
def search_manga():
    try:
        query = request.args.get('q', '').lower()
        if len(query) < 3: return jsonify([])
        results = db_search_titles(query)
        return jsonify(results)
    except Exception: return jsonify([])

@application.route('/api/manga_details', methods=['GET'])
def get_manga_details_route():
    title = request.args.get('title')
    if not title: return jsonify({"error": "Title required"}), 400
    
    data = get_full_manga_data_smart(title)
    
    if data:
        data = {k: (None if pd.isna(v) else v) for k, v in data.items()}
        return jsonify(data)
    return jsonify({"error": "Not found"}), 404

@application.route('/api/chat_recommend', methods=['POST'])
def chat_recommend():
    try:
        data = request.get_json()
        user_manga = data.get('manga')
        user_score = float(data.get('score', 10))
        user_query = data.get('query', '')

        # 1. Initialize History
        if 'history' not in session:
            session['history'] = []

        # 2. Safety Check
        nsfw_triggers = ['hentai', 'erotica', 'porn', 'doujinshi', 'sex', '18+', 'adult']
        is_nsfw = any(t in user_query.lower() for t in nsfw_triggers) or \
                  (user_manga and any(t in user_manga.lower() for t in nsfw_triggers))

        context_data = []
        full_rec_objects = []
        
        # --- DATA GATHERING (Keep your existing logic) ---
        if user_manga:
            # Mode A (SVD) Logic...
            details = db_get_manga_details(user_manga)
            canon_input = details['Title'] if details else user_manga
            rec_titles = get_hybrid_recommendations({canon_input: user_score}, n_recommendations=3)
            input_synop = details.get('Synopsis', "No synopsis.") if details else "No synopsis."
            if is_nsfw: input_synop = "[Explicit content hidden for safety]"
            context_data.append(f"User Input: {user_manga} (Rated {user_score}/10)\nSynopsis: {input_synop}")
            
            for i, title in enumerate(rec_titles):
                d = get_full_manga_data_smart(title)
                if d:
                    full_rec_objects.append(d)
                    safe_synopsis = d.get('Synopsis') if not is_nsfw else "Explicit content matching user request."
                    context_data.append(f"Candidate #{i+1}: {title}\nSynopsis: {safe_synopsis}")
            
            system_instruction = """You are a manga expert. 
            The user likes a specific manga. Recommend the numbered candidates.
            Format titles exactly like: **#1 Title Name**."""
        else:
            # Mode B (Semantic) Logic...
            rec_objects = db_search_by_synopsis(user_query)
            for i, d in enumerate(rec_objects):
                full_rec_objects.append(d)
                safe_synopsis = d.get('Synopsis') if not is_nsfw else "Explicit content matching user request."
                context_data.append(f"Result #{i+1}: {d['Title']}\nSynopsis: {safe_synopsis}")
            
            if is_nsfw:
                system_instruction = "You are a manga librarian. List the numbered results neutrally. Format: **#1 Title Name**."
            else:
                system_instruction = "You are a manga librarian. Explain why these matches fit. Format: **#1 Title Name**. Do not recommend explicit content."

        # --- CONSTRUCT MESSAGE ---
        current_turn_content = f"""
        User Query: "{user_query}"
        Database Matches:
        {"-"*20}
        {chr(10).join(context_data)}
        {"-"*20}
        """

        if not session['history']:
             session['history'].append({"role": "system", "content": system_instruction})
        
        session['history'].append({"role": "user", "content": current_turn_content})

        # --- NEW: MULTI-MODEL BACKUP SYSTEM ---
        
        # Priority List:
        # 1. Llama 3.3 70B (Smartest/Best for Manga reasoning)
        # 2. Qwen 3 32B (Very strong mid-sized model, great backup)
        # 3. Llama 3.1 8B (Instant/Fast, good if the others are busy)
        # 4. Kimi K2 (Good alternative if Llama is down)
        # 5. GPT-OSS 120B (Heavy duty backup)
        available_models = [
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "llama-3.1-8b-instant",
            "moonshotai/kimi-k2-instruct",
            "openai/gpt-oss-120b",
            "groq/compound-mini"
        ]

        llm_response = "I'm having trouble connecting to my brain right now. Please try again later."
        
        if client:
            for model_name in available_models:
                try:
                    print(f"🤖 Attempting to generate with model: {model_name}...")
                    completion = client.chat.completions.create(
                        messages=session['history'], 
                        model=model_name,
                        temperature=0.7,
                        max_tokens=450
                    )
                    llm_response = completion.choices[0].message.content
                    print(f"✅ Success with {model_name}")
                    break # Exit the loop if successful!
                
                except RateLimitError as e:
                    print(f"⚠️ RATE LIMIT HIT on {model_name}. Switching to backup...")
                    continue # Try the next model in the list
                
                except APIError as e:
                    print(f"❌ API Error on {model_name}: {e}")
                    continue # Try next model
                    
                except Exception as e:
                    print(f"🔥 Unexpected Error on {model_name}: {e}")
                    break # If it's a code error, don't keep retrying
        else:
            llm_response = "Server Configuration Error: No API Key found."

        session['history'].append({"role": "assistant", "content": llm_response})

        clean_recs = pd.DataFrame(full_rec_objects).replace({np.nan: None}).to_dict(orient='records')
        return jsonify({"llm_response": llm_response, "recommendations": clean_recs})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@application.route('/api/reset_chat', methods=['POST'])
def reset_chat():
    session.pop('history', None) # Delete the history key
    return jsonify({"status": "success", "message": "Memory wiped!"})

if __name__ == '__main__':
    application.run(host="localhost", port=5000, debug=True)