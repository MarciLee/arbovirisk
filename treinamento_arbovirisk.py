# treinamento_arbovirisk.py v 2.2.0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# 1. Carregar dados
url = "https://zenodo.org/records/17902345/files/Synthetic%20and%20Encoded%20Database%20of%20Dengue,%20Zika,%20Chikungunya,%20and%20Influenza%20Derived%20from%20the%20Literature.xlsx?download=1"
df_dengue = pd.read_excel(url, sheet_name='DENGUE')
df_zika = pd.read_excel(url, sheet_name='ZIKA')
df_chik = pd.read_excel(url, sheet_name='CHIKUNGUNYA')

df_dengue['DIAGNOSTICO'] = 'Dengue'
df_zika['DIAGNOSTICO'] = 'Zika'
df_chik['DIAGNOSTICO'] = 'Chikungunya'
df = pd.concat([df_dengue, df_zika, df_chik], ignore_index=True)

# 2. Lista de sintomas relevantes (13 features)
sintomas_relevantes = [
    'Fever', 'Headache', 'Myalgia', 'Arthralgia', 'Rash',
    'Retro-orbital pain', 'Vomiting', 'Abdominal pain',
    'Conjunctivitis', 'Pruritus', 'Lymphadenopathy',
    'Leucopenia', 'Thrombocytopenia'
]

feature_cols = [col for col in sintomas_relevantes if col in df.columns]
print(f"Features utilizadas ({len(feature_cols)}): {feature_cols}")

# 3. Aplicar pesos aos sintomas (baseado na PAHO/artigos)
pesos = {
    'Fever': 0.8,
    'Headache': 0.4,
    'Myalgia': 0.7,
    'Arthralgia': 0.9,
    'Rash': 0.7,
    'Retro-orbital pain': 0.8,
    'Vomiting': 0.5,
    'Abdominal pain': 0.5,
    'Conjunctivitis': 0.8,
    'Pruritus': 0.6,
    'Lymphadenopathy': 0.4,
    'Leucopenia': 0.6,
    'Thrombocytopenia': 0.9
}

X = df[feature_cols].copy()
y = df['DIAGNOSTICO']

# Aplicar pesos (multiplicar valores binários)
for col in feature_cols:
    if col in pesos:
        X[col] = X[col] * pesos[col]

print(f"Total amostras: {len(df)}")
print(y.value_counts())

# 4. Codificar target
le_disease = LabelEncoder()
y_enc = le_disease.fit_transform(y)

# 5. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

# 6. Normalização
scaler_disease = StandardScaler()
X_train_scaled = scaler_disease.fit_transform(X_train)
X_test_scaled = scaler_disease.transform(X_test)

# 7. SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# 8. Modelo Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)

y_pred = rf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Modelo Random Forest ===")
print(f"Acurácia: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_disease.classes_))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 9. Modelo de risco da dengue (baseado em sintomas de alarme)
sinais_alarme = [
    'Abdominal pain', 'Vomiting', 'Bleeding', 'Mucosal bleeding',
    'Lethargy', 'Hepatomegaly', 'Plasma leakage', 'Shock',
    'Impaired consciousness'
]
df_dengue_only = df[df['DIAGNOSTICO'] == 'Dengue'].copy()
presentes = [s for s in sinais_alarme if s in df_dengue_only.columns]
df_dengue_only['num_alarme'] = df_dengue_only[presentes].sum(axis=1)
df_dengue_only['RISCO'] = (df_dengue_only['num_alarme'] >= 2).astype(int)

X_risk = df_dengue_only[feature_cols]
y_risk = df_dengue_only['RISCO']

X_risk_train, X_risk_test, y_risk_train, y_risk_test = train_test_split(
    X_risk, y_risk, test_size=0.3, random_state=42, stratify=y_risk
)

scaler_risk = StandardScaler()
X_risk_train_scaled = scaler_risk.fit_transform(X_risk_train)
X_risk_test_scaled = scaler_risk.transform(X_risk_test)

smote_risk = SMOTE(random_state=42)
X_risk_train_bal, y_risk_train_bal = smote_risk.fit_resample(X_risk_train_scaled, y_risk_train)

rf_risk = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_risk.fit(X_risk_train_bal, y_risk_train_bal)

y_risk_pred = rf_risk.predict(X_risk_test_scaled)
acc_risk = accuracy_score(y_risk_test, y_risk_pred)
print(f"\n=== Modelo de Risco da Dengue ===")
print(f"Acurácia: {acc_risk:.4f}")
print(classification_report(y_risk_test, y_risk_pred))

# 10. Salvar artefatos
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/modelo_doenca.pkl')
joblib.dump(scaler_disease, 'models/scaler_doenca.pkl')
joblib.dump(le_disease, 'models/label_encoder_doenca.pkl')
joblib.dump(rf_risk, 'models/modelo_risco.pkl')
joblib.dump(scaler_risk, 'models/scaler_risco.pkl')
joblib.dump(feature_cols, 'models/feature_names.pkl')

print("\n✅ Artefatos salvos em 'models/'")