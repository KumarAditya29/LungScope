from app.db.database import engine
from app.models.user import User  # Import User model first
from app.models.medical_scan import MedicalScan  # Then import MedicalScan

def create_medical_scans_table():
    # Create all tables (this will create both users and medical_scans if needed)
    User.metadata.create_all(bind=engine)
    MedicalScan.metadata.create_all(bind=engine)
    print("Medical scans table created successfully!")

if __name__ == "__main__":
    create_medical_scans_table()
