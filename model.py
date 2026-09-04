import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
#data
df = pd.read_csv("Palo Alto Networks.csv")

if df['Attrition'].dtype == object:
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# feature columns
cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
num_cols = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

X = df[cat_cols + num_cols]
y = df['Attrition']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Pipeline definition
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42))
])

# Model Training
model_pipeline.fit(X_train, y_train)

# Model Evaluation
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 50)
print("PALO ALTO NETWORKS - ATTRITION MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy Score : {acc * 100:.2f}%")
print(f"ROC-AUC Score  : {auc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Retained (0)', 'Exited (1)']))

# Save 
joblib.dump(model_pipeline, "attrition_model.pkl")
print("Model pipeline successfully exported to 'attrition_model.pkl'")