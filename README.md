
# README.md
## 1. Problem Description
This project uses Machine Learning to predict the presence of heart disease based on clinical data. It is deployed as a high-performance asynchronous API using **FastAPI**.

## 2. Dataset
We use the Heart Disease Dataset (available on Kaggle).

Source: Kaggle - Heart Disease Dataset
Download: Place heart.csv inside a folder named data/.
## 3. Environment Setup (Using uv)
We use `uv` for fast dependency management.

pip install uv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip sync
## 4. Running the Project
### Step 1: Training
python train.py
### Step 2: Running the API
We use Uvicorn, a lightning-fast ASGI server, to run FastAPI.

uvicorn predict:app --reload
### Step 3: Testing
Visit http://127.0.0.1:8000/docs for the interactive API documentation (Swagger UI).

Or send a request:

curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}'

## 5. Docker Deployment

### Build
``` bash
docker build -t fastapi-heart .
```

### Run
``` bash
docker run -p 5000:5000 fastapi-heart
```

## Appendix - Meaning of the various columns :

### 1. Patient Demographics
*"age"*: 63
Meaning: The age of the patient in years.
Context: Risk factors for heart disease naturally increase with age. 63 is in the high-risk bracket.
*"sex"*: 1
Meaning: Gender.
Context: 1 = Male, 0 = Female. In this dataset, men are statistically more prone to heart disease in this age group.
### 2. Vital Signs (Resting)
*"trestbps"*: 145
Full Name: Resting Blood Pressure (in mm Hg).
Context: A normal reading is below 120.
Interpretation: 145 is Stage 1 Hypertension (High Blood Pressure). The heart is pumping against higher resistance, which thickens the heart muscle over time.
*"chol"*: 233
Full Name: Serum Cholesterol (in mg/dl).
Context: A desirable level is below 200.
Interpretation: 233 is Borderline High. Cholesterol builds up plaque in the arteries (atherosclerosis), restricting blood flow.
### 3. Blood & Sugar Markers
*"fbs"*: 1
Full Name: Fasting Blood Sugar (> 120 mg/dl).
Context: 1 = True (Sugar > 120), 0 = False.
Interpretation: High Blood Sugar. This indicates diabetes or pre-diabetes. High blood sugar damages blood vessels and is a major risk factor for heart attacks.
### 4. Clinical Indicators (Heart Function)
*"cp"* : 3
Full Name: Chest Pain Type.
Context:
1 = Typical Angina (Classical heart pain)
2 = Atypical Angina (Non-standard pain)
3 = Non-Anginal Pain
4 = Asymptomatic
Interpretation: Non-Anginal Pain. This means the patient has symptoms that are not the classic crushing chest pain. In ML models, this is often risky because it leads to delayed diagnosis (patients think it's just heartburn).
*"restecg"* : 0
Full Name: Resting Electrocardiographic Results.
Context: 0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy.
Interpretation: Normal ECG. While a good sign, a resting ECG often misses heart disease that only appears during stress (exercise).
### 5. Stress Test Results (The most predictive features)
*"thalach"* : 150
Full Name: Maximum Heart Rate Achieved (during exercise).
Context: Formula 220 - Age gives an approximate max (220 - 63 = 157).
Interpretation: 150 is decent. The patient was able to reach a reasonably high heart rate, suggesting they don't have severe failure, but they aren't in peak fitness.
*"exang"* : 0
Full Name: Exercise Induced Angina.
Context: 1 = Yes, 0 = No.
Interpretation: No pain during exercise. This is usually a positive sign (no angina), but combined with the next feature (oldpeak), it suggests "Silent Ischemia" (damage without pain).
"oldpeak": 2.3
Full Name: ST Depression Induced by Exercise Relative to Rest.
Context: Measures how much the electrical signal drops during exercise. Normal is < 1.0.
Interpretation: 2.3 is Very High. This is a strong predictor of disease. It means the heart muscle is literally starving for oxygen during stress.
*"slope"* : 0
Full Name: The Slope of the Peak Exercise ST Segment.
Context: Describes the shape of the heart rate recovery curve.
Interpretation: Depending on the encoding, 0 is often "Upsloping" (Healthier) or "Downsloping" (Unhealthier). In this specific dataset, 0 usually maps to "Upsloping". However, the high oldpeak (2.3) overrides this to indicate the patient is still at high risk.
### 6. Imaging & Defects
*"ca"*: 0
Full Name: Number of Major Vessels Colored by Fluoroscopy (0-3).
Context: Doctors inject dye to see blockages.
Interpretation: 0 vessels blocked. This is confusing because it looks healthy, but fluoroscopy only sees major vessels. It misses "microvascular disease" (disease in tiny vessels), which explains why the patient still gets a prediction of 1 despite this "healthy" number.
*"thal"* : 1
Full Name: Thalassemia (Blood Disorder/Stress Test Result).
Context: 0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect.
Interpretation: Fixed Defect. This means there is a specific part of the heart muscle that is permanently damaged (scar tissue from a past silent heart attack) and does not take up blood at all. This is a definitive marker of heart disease.
