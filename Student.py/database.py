from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id              = Column(Integer, primary_key=True, index=True)
    exam_score      = Column(Integer)
    attendance      = Column(Integer)
    submission      = Column(Integer)
    study_hours     = Column(Float)
    cgpa            = Column(Float)
    extracurricular = Column(Integer)
    predicted_tier  = Column(String)
    timestamp       = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)