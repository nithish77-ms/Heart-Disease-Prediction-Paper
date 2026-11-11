from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel, EmailStr, Field
from typing import Literal, Optional
from datetime import datetime, timezone, date
from pathlib import Path
import json, os, random, io, math, re
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from .database import Base, engine, get_db, SessionLocal
from .models import User, Assessment, Goal, Measurement, AuditLog
from .security import hash_password, verify_password, create_token, decode_token
from .model_runtime import heart_model

Base.metadata.create_all(bind=engine)
app=FastAPI(title='Pulse & Pause Heart Health API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('CORS_ORIGINS','http://localhost:5173').split(','), allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

class RegisterIn(BaseModel):
    name:str=Field(min_length=2,max_length=120)
    email:EmailStr
    password:str=Field(min_length=8,max_length=128)
class LoginIn(BaseModel): email:EmailStr; password:str
class ConsentIn(BaseModel): enabled:bool
class AssessmentIn(BaseModel):
    age:Optional[float]=Field(default=None,ge=18,le=120); sex:Optional[int]=Field(default=None,ge=0,le=1); cp:Optional[int]=Field(default=None,ge=0,le=3)
    trestbps:Optional[float]=Field(default=None,ge=50,le=280); chol:Optional[float]=Field(default=None,ge=50,le=800); fbs:Optional[int]=Field(default=None,ge=0,le=1)
    restecg:Optional[int]=Field(default=None,ge=0,le=2); thalach:Optional[float]=Field(default=None,ge=40,le=260); exang:Optional[int]=Field(default=None,ge=0,le=1)
    oldpeak:Optional[float]=Field(default=None,ge=0,le=10); slope:Optional[int]=Field(default=None,ge=0,le=2)
class LifestyleAssessmentIn(BaseModel):
    smoking:Literal['never','former','sometimes','current']
    activity:Literal['low','some','regular']
    diet:Literal['mixed','poor','high_salt','plant_forward']
    sleep_hours:float=Field(ge=0,le=24)
    stress:int=Field(ge=0,le=10)
    systolic:Optional[int]=Field(default=None,ge=50,le=300)
    diastolic:Optional[int]=Field(default=None,ge=30,le=200)
class GoalIn(BaseModel): title:str=Field(min_length=3,max_length=180); category:str; target:int=Field(ge=1,le=100)
class GoalUpdate(BaseModel): progress:int=Field(ge=0,le=1000)
class MeasurementIn(BaseModel): systolic:int=Field(ge=50,le=300); diastolic:int=Field(ge=30,le=200); heart_rate:int=Field(ge=20,le=250); cholesterol:float=Field(ge=50,le=800)
class SimulationIn(BaseModel):
    assessment_id:int
    age:Optional[float]=Field(default=None,ge=18,le=120)
    sex:Optional[int]=Field(default=None,ge=0,le=1)
    cp:Optional[int]=Field(default=None,ge=0,le=3)
    trestbps:Optional[float]=Field(default=None,ge=50,le=280)
    chol:Optional[float]=Field(default=None,ge=50,le=800)
    fbs:Optional[int]=Field(default=None,ge=0,le=1)
    restecg:Optional[int]=Field(default=None,ge=0,le=2)
    thalach:Optional[float]=Field(default=None,ge=40,le=260)
    exang:Optional[int]=Field(default=None,ge=0,le=1)
    oldpeak:Optional[float]=Field(default=None,ge=0,le=10)
    slope:Optional[int]=Field(default=None,ge=0,le=2)

def current_user(authorization: str=Header(default=''), db:Session=Depends(get_db)):
    if not authorization.startswith('Bearer '): raise HTTPException(401,'Please sign in.')
    try: payload=decode_token(authorization[7:]); user_id=int(payload['sub'])
    except Exception: raise HTTPException(401,'Invalid or expired session.')
    user=db.query(User).filter(User.id==user_id).first()
    if not user: raise HTTPException(401,'Account not found.')
    return user
def require_technician(user=Depends(current_user)):
    if user.role!='technician': raise HTTPException(403,'Medical technician role required.')
    return user
def audit(db, actor, action, subject=None, detail=''):
    db.add(AuditLog(actor_id=actor.id, action=action, subject_user_id=subject, detail=detail[:1000])); db.commit()
def user_dict(u): return {'id':u.id,'name':u.name,'email':u.email,'role':u.role,'consent_technician':u.consent_technician,'created_at':u.created_at.isoformat() if u.created_at else None}
def model_output_label(output):
    if int(output)==1:
        return 'The model found patterns that may be associated with a possibility of heart disease.'
    return 'The model did not find patterns associated with a possibility of heart disease.'

def assessment_dict(a):
    inputs=json.loads(a.inputs_json)
    lifestyle=json.loads(a.lifestyle_json)
    output=int(float(a.model_score)>=0.5)
    model_assessment_completed=bool(inputs) and a.class_prediction is not None
    lifestyle_completed=all(
        key in lifestyle
        for key in ('smoking','activity','diet','sleep_hours','stress')
    ) and model_assessment_completed
    return {
        'id':a.id,
        'user_id':a.user_id,
        'created_at':a.created_at.isoformat() if a.created_at else None,
        'inputs':inputs,
        'model_output':model_output_label(output),
        'risk_level':model_output_label(output),
        'model_assessment_completed':model_assessment_completed,
        'lifestyle':lifestyle,
        'lifestyle_assessment_completed':lifestyle_completed,
        'explanation':json.loads(a.explanation_json or '{}'),
        'recommendations':prevention({**inputs,**lifestyle}) if lifestyle_completed else [],
    }

def original_units_lime_condition(condition, input_features, scaler):
    units = {
        'age': 'age (years)',
        'sex': 'sex (dataset code)',
        'cp': 'cp (dataset code)',
        'trestbps': 'resting BP (mmHg)',
        'chol': 'cholesterol (dataset units)',
        'fbs': 'fasting blood sugar (dataset code)',
        'restecg': 'resting ECG (dataset code)',
        'thalach': 'max heart rate (bpm)',
        'exang': 'exercise angina (dataset code)',
        'oldpeak': 'ST depression (dataset units)',
        'slope': 'ST slope (dataset code)',
    }
    number = r'(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+))'

    for index, feature in enumerate(input_features):
        label = units.get(feature, feature)
        minimum = scaler.data_min_[index]
        scale = scaler.data_max_[index] - minimum

        def convert(match, feature_on_left):
            value = minimum + float(match.group('value')) * scale
            rendered = f'{value:.2f}'.rstrip('0').rstrip('.')
            if feature_on_left:
                return f"{feature} {match.group('operator')} {rendered}"
            return f"{rendered} {match.group('operator')} {feature}"

        lower_bound = re.compile(
            rf'(?<![\w.]){number}\s*(?P<operator><=|<|>=|>)\s*{re.escape(feature)}(?![\w])'
        )
        upper_bound = re.compile(
            rf'(?<![\w]){re.escape(feature)}\s*(?P<operator><=|<|>=|>)\s*{number}'
        )
        condition = lower_bound.sub(lambda match: convert(match, False), condition)
        condition = upper_bound.sub(lambda match: convert(match, True), condition)
        condition = re.sub(
            rf'(?<![\w]){re.escape(feature)}(?![\w])',
            label,
            condition,
        )

    condition = re.sub(
        r'(?<![\w])(distance_cluster_\d+)(?![\w])',
        r'\1 (model-derived distance; no clinical unit)',
        condition,
    )
    return condition

def access_assessment(db, user, assessment_id):
    a=db.query(Assessment).filter(Assessment.id==assessment_id).first()
    if not a: raise HTTPException(404,'Assessment not found.')
    if user.role=='user' and a.user_id!=user.id: raise HTTPException(403,'You can only access your own health records.')
    if user.role=='technician':
        patient=db.query(User).filter(User.id==a.user_id).first()
        if not patient or not patient.consent_technician: raise HTTPException(403,'This user has not authorized technician access.')
    return a

def prevention(data):
    rec=[]
    def add(cat, title, body, goal, based_on, goal_target=1):
        rec.append({
            'category':cat,
            'title':title,
            'recommendation':body,
            'weekly_goal':goal,
            'based_on':based_on,
            'goal_target':goal_target,
        })

    diet=data.get('diet')
    if diet in ('high_salt','processed'):
        add(
            'Diet','Make one lower-salt swap',
            'Since you reported a high-salt or processed-food pattern, try swapping one packaged or salty item for a minimally processed option such as fruit, unsalted nuts, beans, or home-prepared food. Compare sodium on labels where available.',
            'Make 3 lower-salt swaps this week',
            'You selected a high-salt / processed diet pattern.',
            3,
        )
    elif diet=='poor':
        add(
            'Diet','Build one balanced meal at a time',
            'Since you said your usual diet is often unbalanced, start with one meal: add a vegetable or fruit and choose a whole-grain or bean option when available. Keep changes affordable and realistic for you.',
            'Add a fruit or vegetable to one meal on 5 days',
            'You described your usual diet as often unbalanced.',
            5,
        )
    elif diet=='plant_forward':
        add(
            'Diet','Keep your plant-forward routine varied',
            'You already report a plant-forward pattern. Keep variety across vegetables, fruit, beans, lentils, whole grains, nuts, and seeds; choose unsaturated fats and limit highly processed foods where practical.',
            'Include beans, lentils, or nuts in 3 meals',
            'You selected a plant-forward diet pattern.',
            3,
        )
    elif diet=='mixed':
        add(
            'Diet','Choose one nourishing addition',
            'You reported a mixed diet. Rather than overhauling it, add a vegetable, fruit, bean, or whole-grain option to a meal you already enjoy.',
            'Add a fruit, vegetable, or bean to one meal on 5 days',
            'You selected a mixed / varied diet pattern.',
            5,
        )
    else:
        add(
            'Diet','Choose a nourishing addition',
            'Try adding a vegetable, fruit, bean, or whole-grain option to a meal you already enjoy. Keep the change practical for your preferences and budget.',
            'Add a fruit, vegetable, or bean to one meal on 5 days',
            'A diet pattern was not recorded, so this is a general suggestion.',
            5,
        )

    activity=data.get('activity')
    if activity in ('low','sedentary','none'):
        add(
            'Physical activity','Start with short, comfortable movement',
            'You reported low or sedentary activity. If it feels safe for you, begin with easy 10-minute walks and build gradually. Stop and seek medical advice for chest discomfort, faintness, or unusual breathlessness.',
            'Try a comfortable 10-minute walk on 3 days',
            'You reported low / sedentary activity.',
            3,
        )
    elif activity=='some':
        add(
            'Physical activity','Add a little to your current activity',
            'You reported doing some activity. Keep your current routine and, if comfortable, add two brisk but manageable 10-minute movement breaks this week. Increase gradually rather than making a sudden jump.',
            'Add two 10-minute movement sessions',
            'You reported doing some activity.',
            2,
        )
    elif activity=='regular':
        add(
            'Physical activity','Maintain the routine that works for you',
            'You reported regular activity. Keep a sustainable schedule with recovery time, and adjust intensity to your comfort and any clinician guidance.',
            'Schedule your usual activity on 4 days',
            'You reported regular activity.',
            4,
        )
    else:
        add(
            'Physical activity','Choose comfortable movement',
            'If it feels safe for you, add short, comfortable movement breaks and increase gradually. Stop and seek medical advice for chest discomfort, faintness, or unusual breathlessness.',
            'Try a comfortable 10-minute movement break on 3 days',
            'An activity level was not recorded, so this is a gentle starting suggestion.',
            3,
        )

    smoking=data.get('smoking')
    if smoking in ('current','daily','sometimes'):
        add(
            'Smoking','Choose a practical tobacco-quit support step',
            'You reported current or occasional smoking. If you are ready, choose one support step such as talking with a clinician or contacting a local evidence-based quit service. Support can help tailor options to you.',
            'Choose one quit-support step',
            'You reported smoking sometimes or currently.',
            1,
        )
    elif smoking=='former':
        add(
            'Smoking','Support your smoke-free progress',
            'You reported that you used to smoke. Recognize the progress you have made and consider what support or trigger plan would help you stay smoke-free.',
            'Write down one strategy that helps you stay smoke-free',
            'You reported being a former smoker.',
            1,
        )
    elif smoking=='never':
        add(
            'Smoking','Protect your smoke-free routine',
            'You reported that you do not smoke. Continue avoiding tobacco smoke where possible and seek support if your circumstances or exposure change.',
            'Note one way to avoid smoke exposure this week',
            'You reported never smoking.',
            1,
        )
    else:
        add(
            'Smoking','Consider tobacco exposure',
            'Avoid tobacco smoke where possible. If you smoke or want help avoiding exposure, a healthcare professional or evidence-based quit service can discuss support options.',
            'Note one tobacco-exposure or support question',
            'Smoking status was not recorded.',
            1,
        )

    try:
        sleep_hours=float(data.get('sleep_hours'))
    except (TypeError,ValueError):
        sleep_hours=None
    if sleep_hours is not None and sleep_hours<7:
        add(
            'Sleep','Make a small change to protect sleep time',
            f'You reported about {sleep_hours:g} hours of sleep per night. If possible, move bedtime earlier by 15 minutes and keep a steady wake time. Talk with a clinician if poor sleep persists or affects daily life.',
            'Try a consistent bedtime on 4 nights',
            f'Your questionnaire reported {sleep_hours:g} hours of sleep per night.',
            4,
        )
    elif sleep_hours is not None and sleep_hours>9:
        add(
            'Sleep','Check in about your sleep pattern',
            f'You reported about {sleep_hours:g} hours of sleep per night. Sleep needs vary; if this is usual for you or you still feel tired, consider discussing sleep quality with a healthcare professional.',
            'Note sleep duration and how rested you feel on 4 days',
            f'Your questionnaire reported {sleep_hours:g} hours of sleep per night.',
            4,
        )
    else:
        add(
            'Sleep','Keep the sleep routine that suits you',
            'You reported around 7–9 hours of sleep per night. Keep a regular schedule and a relaxing wind-down routine if these are working for you.',
            'Keep a regular sleep schedule on 4 nights',
            f'Your questionnaire reported {sleep_hours:g} hours of sleep per night.' if sleep_hours is not None else 'Sleep duration was not recorded.',
            4,
        )

    try:
        stress=int(data.get('stress'))
    except (TypeError,ValueError):
        stress=None
    if stress is not None and stress>=7:
        add(
            'Stress','Pick a short reset that feels manageable',
            f'You rated stress {stress}/10. Choose one brief reset that suits you, such as slow breathing, a short walk, or talking with someone you trust. If stress feels overwhelming, consider professional support.',
            'Try a 5-minute reset on 3 days',
            f'Your self-rated stress was {stress}/10.',
            3,
        )
    else:
        add(
            'Stress','Protect time for recovery',
            f'You rated stress {stress}/10.' if stress is not None else 'No stress rating was recorded.',
            'Schedule one enjoyable or restorative break on 3 days',
            f'Your self-rated stress was {stress}/10.' if stress is not None else 'Stress rating was not recorded.',
            3,
        )
        rec[-1]['recommendation'] += ' Keep a few brief breaks or enjoyable activities in your week; if stress changes or becomes hard to manage, seek support.'

    bp_sys=data.get('systolic')
    bp_dia=data.get('diastolic')
    bp_source='recent blood pressure reading'
    if bp_sys is None:
        bp_sys=data.get('trestbps')
        bp_source='resting blood-pressure value in your assessment'
    try:
        bp_sys=float(bp_sys) if bp_sys is not None else None
    except (TypeError,ValueError):
        bp_sys=None
    try:
        bp_dia=float(bp_dia) if bp_dia is not None else None
    except (TypeError,ValueError):
        bp_dia=None
    if (bp_sys is not None and bp_sys>=130) or (bp_dia is not None and bp_dia>=80):
        reading=' and '.join(
            value for value in (
                f'{bp_sys:g} mmHg systolic' if bp_sys is not None else '',
                f'{bp_dia:g} mmHg diastolic' if bp_dia is not None else '',
            ) if value
        )
        add(
            'Blood pressure','Recheck and discuss this blood-pressure reading',
            f'Your recorded {bp_source} was {reading}. A single reading cannot diagnose a condition. Repeat it after sitting quietly, using a validated cuff if available, and discuss repeated elevated readings with a healthcare professional. Do not change medication based on this app.',
            'Record readings on 2 separate days if available',
            f'Your {bp_source} was {reading}.',
            2,
        )
    elif bp_sys is not None or bp_dia is not None:
        reading=' and '.join(
            value for value in (
                f'{bp_sys:g} mmHg systolic' if bp_sys is not None else '',
                f'{bp_dia:g} mmHg diastolic' if bp_dia is not None else '',
            ) if value
        )
        add(
            'Blood pressure','Keep a record of your blood pressure',
            f'Your recorded {bp_source} was {reading}. One reading is only a snapshot; follow your clinician’s advice about whether and when to measure it again.',
            'Record another reading if recommended by your clinician',
            f'Your {bp_source} was {reading}.',
            1,
        )

    cholesterol=data.get('chol')
    if cholesterol is not None:
        add(
            'Cholesterol','Review this cholesterol value in context',
            f'Your assessment recorded cholesterol as {cholesterol}. The app cannot determine from this value alone what it means for your health. Ask a healthcare professional how it relates to your full lipid profile and personal history.',
            'Write down one question about your lipid results',
            f'Your assessment recorded cholesterol as {cholesterol}.',
            1,
        )

    chest_pain=data.get('cp')
    exertional_pain=data.get('exang')
    if (chest_pain is not None and str(chest_pain)!='3') or str(exertional_pain)=='1':
        add(
            'Symptoms and follow-up','Discuss the symptoms you reported',
            'Your questionnaire indicates a chest-pain pattern or chest discomfort with exercise. Arrange a timely conversation with a healthcare professional. If you have new, severe, persistent chest pain, severe shortness of breath, fainting, or other emergency symptoms, seek emergency care now.',
            'Write down when symptoms happen and what they feel like',
            'Your assessment recorded a chest-pain response that may need clinical context.',
            1,
        )
    else:
        add(
            'Follow-up','Use your results as a conversation aid',
            'The model output is not a diagnosis or a calibrated clinical probability. Discuss questions, symptoms, family history, and preventive screening with a qualified professional.',
            'Plan a routine check-in if one is due',
            'General follow-up applies to all assessment results.',
            1,
        )
    return rec

def make_explanation(X):
    b=heart_model.bundle; features=list(X.columns); result={'features':features,'shap':[],'lime':[],'permutation':[],'pdp_ice':[],'note':'Explanations describe model behaviour, not medical causation.'}
    reference=b.get('training_reference')
    if reference is None or len(reference)==0: return result
    try:
        import shap
        bg=reference.head(min(20,len(reference)))
        def pred(arr): return heart_model.model.predict_proba(pd.DataFrame(arr,columns=features))[:,1]
        explainer=shap.Explainer(pred,bg,algorithm='permutation')
        exp=explainer(X,max_evals=max(2*len(features)+1,25))
        vals=np.asarray(exp.values)
        if vals.ndim==3: vals=vals[:,:,0]
        row=vals[0]
        result['shap']=[{'feature':features[i],'value':float(row[i]),'abs_value':float(abs(row[i]))} for i in np.argsort(np.abs(row))[::-1][:10]]
        global_vals=np.mean(np.abs(vals),axis=0)
        result['shap_global']=[{'feature':features[i],'importance':float(global_vals[i])} for i in np.argsort(global_vals)[::-1][:10]]
    except Exception as e: result['shap_error']=str(e)[:200]
    try:
        from lime.lime_tabular import LimeTabularExplainer
        lx=LimeTabularExplainer(reference.to_numpy(),feature_names=features,class_names=['The model did not find patterns associated with a possibility of heart disease.','The model found patterns that may be associated with a possibility of heart disease.'],mode='classification',discretize_continuous=True,random_state=42)
        ex=lx.explain_instance(X.iloc[0].to_numpy(),heart_model.model.predict_proba,num_features=min(8,len(features)),labels=(1,))
        result['lime']=[{'condition':original_units_lime_condition(str(k),b['input_features'],b['scaler']),'contribution':float(v)} for k,v in ex.as_list(label=1)]
    except Exception as e: result['lime_error']=str(e)[:200]
    try:
        top=[x['feature'] for x in result.get('shap_global',[])[:3]] or features[:3]
        ice=[]
        sample=reference.head(min(25,len(reference)))
        for f in top:
            lo,hi=np.percentile(reference[f],[10,90]); grid=np.linspace(float(lo),float(hi),12)
            curves=[]
            for value in grid:
                changed=sample.copy(); changed[f]=value
                curves.append(heart_model.model.predict_proba(changed)[:,1].tolist())
            ice.append({'feature':f,'grid':[round(float(v),5) for v in grid],'pdp':[float(np.mean(c)) for c in curves],'ice':[[float(curves[j][i]) for j in range(len(grid))] for i in range(len(sample))][:12]})
        result['pdp_ice']=ice
    except Exception as e: result['pdp_ice_error']=str(e)[:200]
    try:
        holdout=b.get('holdout_reference'); labels=b.get('holdout_y')
        if holdout is not None and labels is not None and len(holdout)==len(labels):
            labels=np.asarray(labels)
            baseline=float(np.mean(heart_model.model.predict(holdout)==labels))
            rng=np.random.RandomState(42)
            importances=[]
            for feature in features:
                scores=[]
                for _ in range(5):
                    permuted=holdout.copy()
                    permuted[feature]=rng.permutation(permuted[feature].to_numpy())
                    scores.append(baseline-float(np.mean(heart_model.model.predict(permuted)==labels)))
                importances.append({
                    'feature':feature,
                    'importance':float(np.mean(scores)),
                    'std':float(np.std(scores)),
                })
            result['permutation']=sorted(importances,key=lambda item:item['importance'],reverse=True)[:10]
            result['permutation_note']='Computed on the saved held-out test subset from the uploaded training workflow.'
        else:
            result['permutation_note']='No held-out labeled reference was saved; retrain the model to enable permutation importance.'
    except Exception as e:
        result['permutation_note']='Permutation importance could not be computed: '+str(e)[:150]
    return result

@app.get('/api/health')
def health():
    features = heart_model.bundle['input_features'] if heart_model.ready else []
    return {
        'status':'ok',
        'model_ready':heart_model.ready,
        'model_source':heart_model.bundle.get('training_source', 'trained model') if heart_model.ready else 'not trained',
        'training_required':not heart_model.ready,
        'assessment_features':features,
    }
@app.post('/api/auth/register')
def register(data:RegisterIn, db:Session=Depends(get_db)):
    email=data.email.lower().strip()
    if db.query(User).filter(User.email==email).first(): raise HTTPException(409,'Email already registered.')
    u=User(name=data.name.strip(),email=email,password_hash=hash_password(data.password),role='user',consent_technician=False)
    db.add(u); db.commit(); db.refresh(u)
    return {'token':create_token(u.id,u.role),'user':user_dict(u)}
@app.post('/api/auth/login')
def login(data:LoginIn, db:Session=Depends(get_db)):
    u=db.query(User).filter(User.email==data.email.lower().strip()).first()
    if not u or not verify_password(data.password,u.password_hash): raise HTTPException(401,'Invalid email or password.')
    return {'token':create_token(u.id,u.role),'user':user_dict(u)}
@app.get('/api/me')
def me(user=Depends(current_user)): return user_dict(user)
@app.put('/api/me/technician-consent')
def set_consent(data:ConsentIn,user=Depends(current_user),db:Session=Depends(get_db)):
    if user.role!='user': raise HTTPException(400,'Consent setting is for patient accounts.')
    user.consent_technician=data.enabled; db.commit(); audit(db,user,'technician_consent_changed',user.id,str(data.enabled)); return user_dict(user)
@app.post('/api/assessments')
def create_assessment(data:AssessmentIn,user=Depends(current_user),db:Session=Depends(get_db)):
    if not heart_model.ready: raise HTTPException(503,'The uploaded model has not been trained yet. Add heart.csv and run python train_model.py.')
    raw=data.model_dump()
    feature_names=heart_model.bundle['input_features']
    missing=[feature for feature in feature_names if raw.get(feature) is None]
    if missing: raise HTTPException(422,'Missing required model inputs: '+', '.join(missing))
    model_inputs={k:raw[k] for k in feature_names}
    try: prediction=heart_model.predict(model_inputs)
    except Exception as e: raise HTTPException(500,'Model inference failed: '+str(e)[:180])
    output=prediction['class_prediction']
    score=prediction['model_score']
    risk='possible_heart_disease' if output==1 else 'no_pattern_found'
    X=pd.DataFrame([prediction['selected_features']])
    xai=make_explanation(X)
    a=Assessment(user_id=user.id,inputs_json=json.dumps(model_inputs),model_score=score,class_prediction=output,risk_level=risk,lifestyle_json=json.dumps({}),explanation_json=json.dumps(xai),model_version='ISSO-MC-RF-SVM-KNN-MLP-HSO')
    db.add(a); db.commit(); db.refresh(a); audit(db,user,'assessment_created',user.id,f'assessment_id={a.id}')
    return assessment_dict(a)
@app.put('/api/assessments/{assessment_id}/lifestyle')
def save_lifestyle_assessment(assessment_id:int,data:LifestyleAssessmentIn,user=Depends(current_user),db:Session=Depends(get_db)):
    if user.role!='user': raise HTTPException(403,'Only personal user accounts can complete a lifestyle assessment.')
    assessment=db.query(Assessment).filter(
        Assessment.id==assessment_id,
        Assessment.user_id==user.id,
    ).first()
    if not assessment: raise HTTPException(404,'Assessment not found.')
    assessment.lifestyle_json=json.dumps(data.model_dump())
    db.commit()
    db.refresh(assessment)
    audit(db,user,'lifestyle_assessment_completed',user.id,f'assessment_id={assessment.id}')
    return assessment_dict(assessment)
@app.get('/api/assessments')
def list_assessments(user=Depends(current_user),db:Session=Depends(get_db)):
    q=db.query(Assessment)
    if user.role=='user': q=q.filter(Assessment.user_id==user.id)
    else: q=q.join(User,Assessment.user_id==User.id).filter(User.consent_technician==True)
    return [assessment_dict(a) for a in q.order_by(desc(Assessment.created_at)).limit(200).all()]
@app.get('/api/assessments/{assessment_id}')
def get_assessment(assessment_id:int,user=Depends(current_user),db:Session=Depends(get_db)):
    return assessment_dict(access_assessment(db,user,assessment_id))
@app.get('/api/technician/users')
def technician_users(user=Depends(require_technician),db:Session=Depends(get_db)):
    users=db.query(User).filter(User.role=='user',User.consent_technician==True).order_by(User.name).all()
    audit(db,user,'view_authorized_patient_list',None,f'count={len(users)}')
    return [user_dict(u) for u in users]
@app.get('/api/technician/users/{user_id}/assessments')
def technician_user_assessments(user_id:int,user=Depends(require_technician),db:Session=Depends(get_db)):
    patient=db.query(User).filter(User.id==user_id,User.role=='user',User.consent_technician==True).first()
    if not patient: raise HTTPException(403,'Patient not found or has not authorized technician access.')
    audit(db,user,'view_patient_assessments',user_id)
    return [assessment_dict(a) for a in db.query(Assessment).filter(Assessment.user_id==user_id).order_by(desc(Assessment.created_at)).limit(100).all()]
@app.post('/api/goals')
def create_goal(data:GoalIn,user=Depends(current_user),db:Session=Depends(get_db)):
    g=Goal(user_id=user.id,title=data.title,category=data.category,target=data.target,progress=0,week_start=date.today().isoformat(),completed=False)
    db.add(g); db.commit(); db.refresh(g); return goal_dict(g)
def goal_dict(g): return {'id':g.id,'title':g.title,'category':g.category,'target':g.target,'progress':g.progress,'week_start':g.week_start,'completed':g.completed}
@app.get('/api/goals')
def list_goals(user=Depends(current_user),db:Session=Depends(get_db)):
    return [goal_dict(g) for g in db.query(Goal).filter(Goal.user_id==user.id).order_by(desc(Goal.id)).limit(100).all()]
@app.patch('/api/goals/{goal_id}')
def update_goal(goal_id:int,data:GoalUpdate,user=Depends(current_user),db:Session=Depends(get_db)):
    g=db.query(Goal).filter(Goal.id==goal_id,Goal.user_id==user.id).first()
    if not g: raise HTTPException(404,'Goal not found.')
    g.progress=data.progress; g.completed=data.progress>=g.target; db.commit(); db.refresh(g); return goal_dict(g)
@app.post('/api/simulations')
def simulate_assessment(data:SimulationIn,user=Depends(current_user),db:Session=Depends(get_db)):
    if user.role!='user': raise HTTPException(403,'Only personal user accounts can run simulations.')
    if not heart_model.ready: raise HTTPException(503,'The uploaded model has not been trained yet.')
    assessment=db.query(Assessment).filter(
        Assessment.id==data.assessment_id,
        Assessment.user_id==user.id,
    ).first()
    if not assessment: raise HTTPException(404,'Baseline assessment not found.')

    stored_inputs=json.loads(assessment.inputs_json)
    features=heart_model.bundle['input_features']
    missing=[feature for feature in features if feature not in stored_inputs]
    if missing: raise HTTPException(409,'This assessment does not contain the model inputs required for simulation.')
    changed_inputs=data.model_dump(exclude={'assessment_id'},exclude_unset=True)
    unsupported=[feature for feature in changed_inputs if feature not in features]
    if unsupported: raise HTTPException(422,'Only selected model features can be simulated: '+', '.join(unsupported))
    if not changed_inputs: raise HTTPException(422,'Change at least one selected measurement to run a simulation.')
    if any(value is None for value in changed_inputs.values()): raise HTTPException(422,'Simulation values cannot be empty.')
    baseline_inputs={feature:stored_inputs[feature] for feature in features}
    model_inputs={**baseline_inputs,**changed_inputs}
    try:
        baseline_prediction=heart_model.predict(baseline_inputs)['class_prediction']
        simulated_output=heart_model.predict(model_inputs)['class_prediction']
    except Exception as e:
        raise HTTPException(500,'Simulation inference failed: '+str(e)[:180])

    return {
        'baseline_output':model_output_label(baseline_prediction),
        'simulated_output':model_output_label(simulated_output),
        'output_label':model_output_label(simulated_output),
        'changed_measurements':changed_inputs,
        'saved':False,
        'note':'Hypothetical model scenario only; not a clinical prediction or diagnosis.',
    }
@app.post('/api/measurements/simulate')
def simulate_measurement(user=Depends(current_user),db:Session=Depends(get_db)):
    # Clearly labelled demonstration data; never represented as real patient readings.
    m=Measurement(user_id=user.id,systolic=random.randint(108,154),diastolic=random.randint(68,98),heart_rate=random.randint(58,102),cholesterol=round(random.uniform(155,255),1),source='synthetic',label='SYNTHETIC — NOT A REAL MEASUREMENT')
    db.add(m); db.commit(); db.refresh(m); return measurement_dict(m)
def measurement_dict(m): return {'id':m.id,'created_at':m.created_at.isoformat() if m.created_at else None,'systolic':m.systolic,'diastolic':m.diastolic,'heart_rate':m.heart_rate,'cholesterol':m.cholesterol,'source':m.source,'label':m.label}
@app.post('/api/measurements')
def add_measurement(data:MeasurementIn,user=Depends(current_user),db:Session=Depends(get_db)):
    m=Measurement(user_id=user.id,**data.model_dump(),source='user_entered',label='USER-ENTERED — NOT VERIFIED')
    db.add(m); db.commit(); db.refresh(m); return measurement_dict(m)
@app.get('/api/measurements')
def list_measurements(user=Depends(current_user),db:Session=Depends(get_db)):
    return [measurement_dict(m) for m in db.query(Measurement).filter(Measurement.user_id==user.id).order_by(desc(Measurement.created_at)).limit(100).all()]
@app.get('/api/reports/{assessment_id}.pdf')
def download_report(assessment_id:int,user=Depends(current_user),db:Session=Depends(get_db)):
    a=access_assessment(db,user,assessment_id); owner=db.query(User).filter(User.id==a.user_id).first()
    buff=io.BytesIO(); doc=SimpleDocTemplate(buff,pagesize=letter,rightMargin=40,leftMargin=40,topMargin=40,bottomMargin=40); styles=getSampleStyleSheet(); story=[]
    story += [Paragraph('Heart Health Assessment & Prevention Summary',styles['Title']),Paragraph('Educational decision-support report — not a diagnosis or a substitute for professional care.',styles['BodyText']),Spacer(1,12)]
    report_inputs=json.loads(a.inputs_json)
    story += [Paragraph(f'Patient: {owner.name if owner else "Account"}',styles['Heading2']),Paragraph(f'Assessment date: {a.created_at.isoformat() if a.created_at else "N/A"}',styles['BodyText']),Paragraph(f'Assessment result: {assessment_dict(a)["model_output"]} This research-model result is not a diagnosis or a calibrated clinical probability.',styles['BodyText']),Spacer(1,10)]
    rows=[['Input','Value']]+[[k,str(v)] for k,v in json.loads(a.inputs_json).items()]
    table=Table(rows,colWidths=[210,270]); table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef7')),('GRID',(0,0),(-1,-1),.4,colors.grey),('PADDING',(0,0),(-1,-1),6)])); story += [table,Spacer(1,12),Paragraph('Personalized prevention ideas',styles['Heading2'])]
    report_recommendations=prevention({**report_inputs,**json.loads(a.lifestyle_json)}) if assessment_dict(a)['lifestyle_assessment_completed'] else []
    if report_recommendations:
        for r in report_recommendations: story.append(Paragraph(f'<b>{r["category"]}: {r["title"]}</b><br/>{r["recommendation"]}<br/><i>Why suggested: {r["based_on"]}</i><br/><i>Weekly goal: {r["weekly_goal"]}</i>',styles['BodyText'])); story.append(Spacer(1,7))
    else:
        story.append(Paragraph('Complete the separate lifestyle assessment in the Prevention Plan tab to generate personalized recommendations.',styles['BodyText']))
    story += [Spacer(1,8),Paragraph('Urgent symptoms such as chest pain, severe shortness of breath, fainting or stroke symptoms require emergency care. Do not change medicines based on this report.',styles['BodyText'])]
    doc.build(story); buff.seek(0); from fastapi.responses import StreamingResponse
    audit(db,user,'download_report',a.user_id,f'assessment_id={a.id}')
    return StreamingResponse(buff,media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename=heart-health-report-{a.id}.pdf'})
@app.get('/api/dashboard')
def dashboard(user=Depends(current_user),db:Session=Depends(get_db)):
    if user.role=='user':
        assessments=db.query(Assessment).filter(Assessment.user_id==user.id).order_by(desc(Assessment.created_at)).limit(12).all()
        measurements=db.query(Measurement).filter(Measurement.user_id==user.id).order_by(desc(Measurement.created_at)).limit(12).all()
        goals=db.query(Goal).filter(Goal.user_id==user.id).all()
        return {'assessments':[assessment_dict(a) for a in assessments],'measurements':[measurement_dict(m) for m in measurements],'goals':[goal_dict(g) for g in goals]}
    users=db.query(User).filter(User.role=='user',User.consent_technician==True).all()
    ids=[u.id for u in users]
    assessments=db.query(Assessment).filter(Assessment.user_id.in_(ids)).order_by(desc(Assessment.created_at)).limit(30).all() if ids else []
    audit(db,user,'technician_dashboard_view',None,f'authorized_users={len(users)}')
    return {'authorized_users':[user_dict(u) for u in users],'assessments':[assessment_dict(a) for a in assessments]}
@app.get('/')
def root(): return {'message':'Heart Prevention API','docs':'/docs','model_ready':heart_model.ready}
