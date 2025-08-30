from app.db.database import engine
from app.models.user import User
from app.models.medical_scan import MedicalScan
from app.models.patient_profile import PatientProfile, MedicalNote

def create_patient_tables():
    """Create patient profile and medical notes tables."""
    PatientProfile.metadata.create_all(bind=engine)
    MedicalNote.metadata.create_all(bind=engine)
    print("Patient tables created successfully!")
    print("- patient_profiles table")
    print("- medical_notes table")

if __name__ == "__main__":
    create_patient_tables()
