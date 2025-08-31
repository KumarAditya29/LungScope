from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from app.db.database import Base
from datetime import datetime

class MedicalScan(Base):
    __tablename__ = "medical_scans"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    scan_type = Column(String, nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    analysis_status = Column(String, default="pending")
