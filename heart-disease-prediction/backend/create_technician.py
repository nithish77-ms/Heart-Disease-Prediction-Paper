import argparse, getpass
from app.database import Base, engine, SessionLocal
from app.models import User
from app.security import hash_password
Base.metadata.create_all(bind=engine)
p=argparse.ArgumentParser(description='Create a medical technician account (run from backend folder).')
p.add_argument('--name',required=True);p.add_argument('--email',required=True)
a=p.parse_args(); password=getpass.getpass('Technician password (minimum 8 characters): ')
if len(password)<8: raise SystemExit('Password must be at least 8 characters.')
db=SessionLocal()
try:
    email=a.email.strip().lower()
    if db.query(User).filter(User.email==email).first(): raise SystemExit('That email is already registered.')
    user=User(name=a.name.strip(),email=email,password_hash=hash_password(password),role='technician',consent_technician=False)
    db.add(user);db.commit();print(f'Technician account created: {email}')
finally: db.close()
