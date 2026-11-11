from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .database import Base
class User(Base):
    __tablename__='users'
    id=Column(Integer, primary_key=True, index=True)
    name=Column(String(120), nullable=False)
    email=Column(String(255), unique=True, index=True, nullable=False)
    password_hash=Column(String(300), nullable=False)
    role=Column(String(30), nullable=False, default='user')
    created_at=Column(DateTime, default=lambda: datetime.now(timezone.utc))
    consent_technician=Column(Boolean, default=False, nullable=False)
    assessments=relationship('Assessment', back_populates='user')
class Assessment(Base):
    __tablename__='assessments'
    id=Column(Integer, primary_key=True, index=True)
    user_id=Column(Integer, ForeignKey('users.id'), index=True, nullable=False)
    created_at=Column(DateTime, default=lambda: datetime.now(timezone.utc))
    inputs_json=Column(Text, nullable=False)
    model_score=Column(Float, nullable=False)
    class_prediction=Column(Integer, nullable=False)
    risk_level=Column(String(40), nullable=False)
    lifestyle_json=Column(Text, nullable=False)
    explanation_json=Column(Text, nullable=True)
    model_version=Column(String(120), default='uploaded-model-artifact')
    user=relationship('User', back_populates='assessments')
class Goal(Base):
    __tablename__='weekly_goals'
    id=Column(Integer, primary_key=True)
    user_id=Column(Integer, ForeignKey('users.id'), index=True, nullable=False)
    title=Column(String(180), nullable=False)
    category=Column(String(80), nullable=False)
    target=Column(Integer, default=1)
    progress=Column(Integer, default=0)
    week_start=Column(String(20), nullable=False)
    completed=Column(Boolean, default=False)
class Measurement(Base):
    __tablename__='measurements'
    id=Column(Integer, primary_key=True)
    user_id=Column(Integer, ForeignKey('users.id'), index=True, nullable=False)
    created_at=Column(DateTime, default=lambda: datetime.now(timezone.utc))
    systolic=Column(Integer, nullable=False)
    diastolic=Column(Integer, nullable=False)
    heart_rate=Column(Integer, nullable=False)
    cholesterol=Column(Float, nullable=False)
    source=Column(String(30), default='synthetic')
    label=Column(String(120), default='SYNTHETIC — NOT A REAL MEASUREMENT')
class AuditLog(Base):
    __tablename__='audit_logs'
    id=Column(Integer, primary_key=True)
    actor_id=Column(Integer, nullable=False)
    action=Column(String(100), nullable=False)
    subject_user_id=Column(Integer, nullable=True)
    detail=Column(Text, default='')
    created_at=Column(DateTime, default=lambda: datetime.now(timezone.utc))
