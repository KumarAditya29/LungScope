from sqlalchemy import Column, Integer, String, DateTime, Text, Date, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base
from datetime import datetime

class PatientProfile(Base):
    __tablename__ = "patient_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Personal Information
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(String, nullable=True)  # "male", "female", "other"
    phone_number = Column(String, nullable=True)
    address = Column(Text, nullable=True)
    
    # Medical Information
    blood_type = Column(String, nullable=True)
    allergies = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    medical_conditions = Column(Text, nullable=True)
    emergency_contact_name = Column(String, nullable=True)
    emergency_contact_phone = Column(String, nullable=True)
    
    # System fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MedicalNote(Base):
    __tablename__ = "medical_notes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    scan_id = Column(Integer, ForeignKey("medical_scans.id"), nullable=True)
    
    note_type = Column(String, nullable=False)  # "doctor_note", "patient_note", "analysis"
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    severity = Column(String, default="normal")  # "normal", "warning", "critical"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=True)  # doctor name or "system"
