from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from pathlib import Path
DB_PATH = Path(__file__).resolve().parents[1] / 'heart_app.db'
engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()
