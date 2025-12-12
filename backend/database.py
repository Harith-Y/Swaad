from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./swaad.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    flavor_profile = Column(JSON, default=None)  # Stores UserProfile as JSON
    favorite_dishes = Column(JSON, default=[])  # List of DishInput objects
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class YelpRawResponse(Base):
    __tablename__ = "yelp_raw_responses"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, index=True)
    endpoint = Column(String, index=True)
    request_params = Column(JSON, default=None)
    response_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

