from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.user import User
from app.models.medical_scan import MedicalScan
from app.models.patient_profile import PatientProfile, MedicalNote
from app.utils.utils import hash_password, verify_password, create_access_token, get_current_user
import os
from datetime import datetime, date
from typing import List, Optional

router = APIRouter()

class UserRegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class MedicalScanResponse(BaseModel):
    id: int
    filename: str
    file_size: int
    scan_type: str
    upload_date: datetime
    analysis_status: str

class PatientProfileRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    blood_type: Optional[str] = None
    allergies: Optional[str] = None
    current_medications: Optional[str] = None
    medical_conditions: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None

class MedicalNoteRequest(BaseModel):
    scan_id: Optional[int] = None
    note_type: str = "patient_note"
    title: str
    content: str
    severity: str = "normal"

# EXISTING ENDPOINTS (UNCHANGED)
@router.post("/register")
async def register_user(user: UserRegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_pw = hash_password(user.password)
    new_user = User(username=user.username, email=user.email, password=hashed_pw)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": f"User '{user.username}' registered successfully!"}

@router.post("/login", response_model=TokenResponse)
async def login_user(login_data: UserLoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == login_data.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    if not verify_password(login_data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me")
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email
    }

@router.get("/test")
async def test_endpoint():
    return {"message": "Test endpoint working!"}

@router.post("/upload-scan")
async def upload_medical_scan(
    file: UploadFile = File(...),
    scan_type: str = "chest_xray",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images allowed")
    
    upload_dir = "uploads/medical_scans"
    os.makedirs(upload_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_user.id}_{timestamp}_{file.filename}"
    file_path = os.path.join(upload_dir, filename)
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    medical_scan = MedicalScan(
        user_id=current_user.id,
        filename=filename,
        file_path=file_path,
        file_size=len(content),
        scan_type=scan_type,
        upload_date=datetime.now(),
        analysis_status="pending"
    )
    
    db.add(medical_scan)
    db.commit()
    db.refresh(medical_scan)
    
    return {
        "message": "Medical scan uploaded successfully",
        "scan_id": medical_scan.id,
        "filename": filename,
        "file_size": len(content),
        "scan_type": scan_type,
        "user_id": current_user.id
    }

@router.get("/my-scans", response_model=List[MedicalScanResponse])
async def get_my_medical_scans(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scans = db.query(MedicalScan).filter(MedicalScan.user_id == current_user.id).order_by(MedicalScan.upload_date.desc()).all()
    
    if not scans:
        return []
    
    return [
        MedicalScanResponse(
            id=scan.id,
            filename=scan.filename,
            file_size=scan.file_size,
            scan_type=scan.scan_type or "unknown",
            upload_date=scan.upload_date,
            analysis_status=scan.analysis_status
        )
        for scan in scans
    ]

@router.get("/scan/{scan_id}")
async def get_scan_details(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scan = db.query(MedicalScan).filter(
        MedicalScan.id == scan_id,
        MedicalScan.user_id == current_user.id
    ).first()
    
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    return {
        "id": scan.id,
        "filename": scan.filename,
        "file_size": scan.file_size,
        "scan_type": scan.scan_type,
        "upload_date": scan.upload_date,
        "analysis_status": scan.analysis_status,
        "file_path": scan.file_path
    }

@router.delete("/scan/{scan_id}")
async def delete_medical_scan(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scan = db.query(MedicalScan).filter(
        MedicalScan.id == scan_id,
        MedicalScan.user_id == current_user.id
    ).first()
    
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if os.path.exists(scan.file_path):
        os.remove(scan.file_path)
    
    db.delete(scan)
    db.commit()
    
    return {"message": f"Medical scan '{scan.filename}' deleted successfully"}

@router.get("/dashboard")
async def get_user_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    total_scans = db.query(MedicalScan).filter(MedicalScan.user_id == current_user.id).count()
    pending_analysis = db.query(MedicalScan).filter(
        MedicalScan.user_id == current_user.id,
        MedicalScan.analysis_status == "pending"
    ).count()
    
    recent_scans = db.query(MedicalScan).filter(
        MedicalScan.user_id == current_user.id
    ).order_by(MedicalScan.upload_date.desc()).limit(5).all()
    
    return {
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email
        },
        "stats": {
            "total_scans": total_scans,
            "pending_analysis": pending_analysis,
            "recent_uploads": len(recent_scans)
        },
        "recent_scans": [
            {
                "id": scan.id,
                "filename": scan.filename,
                "scan_type": scan.scan_type,
                "upload_date": scan.upload_date,
                "analysis_status": scan.analysis_status
            }
            for scan in recent_scans
        ]
    }

# NEW PATIENT PROFILE ENDPOINTS
@router.post("/patient-profile")
async def create_or_update_patient_profile(
    profile_data: PatientProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create or update patient profile."""
    existing_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    
    if existing_profile:
        # Update existing profile
        for field, value in profile_data.dict(exclude_unset=True).items():
            setattr(existing_profile, field, value)
        existing_profile.updated_at = datetime.now()
        db.commit()
        db.refresh(existing_profile)
        return {"message": "Patient profile updated successfully"}
    else:
        # Create new profile
        new_profile = PatientProfile(
            user_id=current_user.id,
            **profile_data.dict(exclude_unset=True)
        )
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        return {"message": "Patient profile created successfully"}

@router.get("/patient-profile")
async def get_patient_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's patient profile."""
    profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    
    return {
        "id": profile.id,
        "first_name": profile.first_name,
        "last_name": profile.last_name,
        "date_of_birth": profile.date_of_birth,
        "gender": profile.gender,
        "phone_number": profile.phone_number,
        "address": profile.address,
        "blood_type": profile.blood_type,
        "allergies": profile.allergies,
        "current_medications": profile.current_medications,
        "medical_conditions": profile.medical_conditions,
        "emergency_contact_name": profile.emergency_contact_name,
        "emergency_contact_phone": profile.emergency_contact_phone,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at
    }

@router.post("/medical-notes")
async def add_medical_note(
    note_data: MedicalNoteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a medical note."""
    new_note = MedicalNote(
        user_id=current_user.id,
        scan_id=note_data.scan_id,
        note_type=note_data.note_type,
        title=note_data.title,
        content=note_data.content,
        severity=note_data.severity,
        created_by=current_user.username
    )
    
    db.add(new_note)
    db.commit()
    db.refresh(new_note)
    
    return {
        "message": "Medical note added successfully",
        "note_id": new_note.id
    }

@router.get("/medical-notes")
async def get_medical_notes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all medical notes for current user."""
    notes = db.query(MedicalNote).filter(
        MedicalNote.user_id == current_user.id
    ).order_by(MedicalNote.created_at.desc()).all()
    
    return [
        {
            "id": note.id,
            "scan_id": note.scan_id,
            "note_type": note.note_type,
            "title": note.title,
            "content": note.content,
            "severity": note.severity,
            "created_at": note.created_at,
            "created_by": note.created_by
        }
        for note in notes
    ]

@router.get("/medical-timeline")
async def get_medical_timeline(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get complete medical timeline - scans and notes combined."""
    scans = db.query(MedicalScan).filter(MedicalScan.user_id == current_user.id).all()
    notes = db.query(MedicalNote).filter(MedicalNote.user_id == current_user.id).all()
    
    timeline = []
    
    # Add scans to timeline
    for scan in scans:
        timeline.append({
            "type": "scan",
            "id": scan.id,
            "title": f"{scan.scan_type.replace('_', ' ').title()} - {scan.filename}",
            "date": scan.upload_date,
            "details": {
                "scan_type": scan.scan_type,
                "file_size": scan.file_size,
                "analysis_status": scan.analysis_status
            }
        })
    
    # Add notes to timeline
    for note in notes:
        timeline.append({
            "type": "note",
            "id": note.id,
            "title": note.title,
            "date": note.created_at,
            "details": {
                "note_type": note.note_type,
                "severity": note.severity,
                "content": note.content[:100] + "..." if len(note.content) > 100 else note.content,
                "created_by": note.created_by
            }
        })
    
    # Sort by date (newest first)
    timeline.sort(key=lambda x: x["date"], reverse=True)
    
    return {
        "timeline": timeline,
        "total_items": len(timeline)
    }
