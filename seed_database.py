import pandas as pd
import pickle
from sqlalchemy import create_engine, text, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base 
from pgvector.sqlalchemy import Vector
import os
from tqdm import tqdm # Import tqdm for the progress bar

# --- 1. DATABASE CONNECTION SETUP ---
# Using the complete database URL provided.
# The 'postgresql+psycopg2://' part tells SQLAlchemy how to connect.
DATABASE_URL = "postgresql://mangadb_z6qn_user:D5HZWxfDzqIaP9UDRWHUCiCBr7ZwCXZB@dpg-d3kjah95pdvs739jfro0-a.ohio-postgres.render.com/mangadb_z6qn"

try:
    engine = create_engine(DATABASE_URL)
    # Test the connection to fail early if credentials are wrong
    with engine.connect() as connection:
        print("✅ Successfully connected to the PostgreSQL database!")
except Exception as e:
    print(f"❌ Failed to connect to the database. Please check your credentials and connection string.")
    print(f"Error details: {e}")
    exit() # Stop the script if connection fails

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 2. DEFINE THE MANGA TABLE SCHEMA ---
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
    embedding = Column(Vector(384)) # Vector column for pgvector

def seed_data():
    """
    A one-time script to load data from pickle/CSV files into the PostgreSQL database.
    """
    db_session = SessionLocal()
    
    try:
        # --- SETUP STEPS ---
        print("--- Ensuring 'vector' extension is enabled in the database... ---")
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        print("--- ✅ 'vector' extension is enabled. ---")

        print("--- Creating table in the database (if it doesn't exist)... ---")
        Base.metadata.create_all(bind=engine)
        
        print("--- Creating vector index (IVFFlat)... ---")
        with engine.connect() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_manga_embedding ON manga USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"))
            conn.commit()

        # --- DATA LOADING AND PREPARATION ---
        print("--- Loading data from local files... ---")
        with open('manga_metadata.pkl', 'rb') as f:
            manga_meta = pickle.load(f)
        with open('embeddings.pkl', 'rb') as f:
            embeddings_data = pickle.load(f)
        
        embedding_map = {title: emb for title, emb in zip(embeddings_data['titles'], embeddings_data['embeddings'])}

        # --- FIX: Fetch existing titles from DB once for efficiency ---
        print("--- Fetching existing manga titles from the database to prevent duplicates... ---")
        existing_titles_query = db_session.query(Manga.title).all()
        existing_titles = {title[0] for title in existing_titles_query}
        print(f"--- Found {len(existing_titles)} existing records. ---")
        # --- END FIX ---

        print(f"--- Preparing to insert new records... ---")
        
        manga_objects_to_add = []
        # --- FIX: Add a set to track titles processed in this specific run ---
        titles_in_current_batch = set()
        # --- END FIX ---

        # Add tqdm here to show progress on the preparation step itself
        for _, row in tqdm(manga_meta.iterrows(), total=len(manga_meta), desc="Preparing Records"):
            title = row['Title']
            
            # --- MODIFIED: Check against DB records AND records in this new batch ---
            if title in existing_titles or title in titles_in_current_batch:
                continue
            # --- END MODIFIED ---

            embedding = embedding_map.get(title)

            if embedding is not None:
                new_manga = Manga(
                    title=title,
                    english_title=row.get('English Title'),
                    synopsis=row.get('Synopsis'),
                    genres=row.get('Genres'),
                    themes=row.get('Themes'),
                    score=row.get('Score'),
                    thumbnail=row.get('Image URL'),
                    url=row.get('URL'),
                    embedding=embedding
                )
                manga_objects_to_add.append(new_manga)
                # --- FIX: Add the title to our set for this run ---
                titles_in_current_batch.add(title)
                # --- END FIX ---


        # --- BATCH INSERT WITH PROGRESS BAR ---
        if manga_objects_to_add:
            print(f"--- Inserting {len(manga_objects_to_add)} new records in batches... ---")
            batch_size = 100

            for i in tqdm(range(0, len(manga_objects_to_add), batch_size), desc="Seeding Database", mininterval=0.1):
                batch = manga_objects_to_add[i:i + batch_size]
                db_session.bulk_save_objects(batch)
                db_session.commit()

            print("\n--- ✅ Data insertion complete! ---")
        else:
            print("--- No new records to insert. Database is likely up to date. ---")

    except Exception as e:
        print(f"An error occurred: {e}")
        db_session.rollback()
    finally:
        db_session.close()

if __name__ == "__main__":
    seed_data()

