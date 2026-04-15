# treinar_arbovirisk_corrigido.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, f1_score
import joblib
import os

# 1. Carregar dados
url = "https://zenodo.org/records/17902345/files/Synthetic%20and%20Encoded%20Database%20of%20Dengue,%20Zika,%20Chikungunya,%20and%20Influenza%20Derived%20from%20the%20Literature.xlsx?download=1"
df_dengue = pd.read_excel(url, sheet_name='DENGUE')
df_zika = pd.read_excel(url, sheet_name='ZIKA')
df_chik = pd.read_excel(url, sheet_name='CHIKUNGUNYA')
df_flu = pd.read_excel(url, sheet_name='INFLUENZA')
df_dengue['DIAGNOSTICO'] = 'Dengue'
df_zika['DIAGNOSTICO'] = 'Zika'
df_chik['DIAGNOSTICO'] = 'Chikungunya'
df_flu['DIAGNOSTICO'] = 'Influenza'
df = pd.concat([df_dengue, df_zika, df_chik, df_flu], ignore_index=True)

# 2. Selecionar apenas colunas numéricas (features)
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[feature_cols]
y = df['DIAGNOSTICO']

print(f"Features numéricas: {len(feature_cols)}")
print(f"Total amostras: {len(df)}")

# 3. Codificar target
le_disease = LabelEncoder()
y_enc = le_disease.fit_transform(y)

# 4. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42, stratify=y_enc)

# 5. Normalização
scaler_disease = StandardScaler()
X_train_scaled = scaler_disease.fit_transform(X_train)
X_test_scaled = scaler_disease.transform(X_test)

# 6. SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# 7. Modelos
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)
xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='mlogloss')
xgb.fit(X_train_bal, y_train_bal)

y_pred_rf = rf.predict(X_test_scaled)
y_pred_xgb = xgb.predict(X_test_scaled)

print("\n=== Modelo 1 - Doença ===")
print(f"RF - Acurácia: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"XGB - Acurácia: {accuracy_score(y_test, y_pred_xgb):.4f}")

rf_f1 = f1_score(y_test, y_pred_rf, average='macro')
xgb_f1 = f1_score(y_test, y_pred_xgb, average='macro')
best_disease = xgb if xgb_f1 >= rf_f1 else rf

# 8. Modelo de risco (apenas dengue) - usando as mesmas features numéricas
df_dengue_only = df[df['DIAGNOSTICO'] == 'Dengue'].copy()
# Simular coluna de risco (para demonstração)
np.random.seed(42)
df_dengue_only['RISCO'] = np.random.choice([0,1], size=len(df_dengue_only), p=[0.7,0.3])
X_risk = df_dengue_only[feature_cols]   # mesmas features
y_risk = df_dengue_only['RISCO']

X_risk_train, X_risk_test, y_risk_train, y_risk_test = train_test_split(X_risk, y_risk, test_size=0.3, random_state=42, stratify=y_risk)
scaler_risk = StandardScaler()
X_risk_train_scaled = scaler_risk.fit_transform(X_risk_train)
X_risk_test_scaled = scaler_risk.transform(X_risk_test)
X_risk_train_bal, y_risk_train_bal = SMOTE(random_state=42).fit_resample(X_risk_train_scaled, y_risk_train)

rf_risk = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_risk.fit(X_risk_train_bal, y_risk_train_bal)
xgb_risk = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='logloss')
xgb_risk.fit(X_risk_train_bal, y_risk_train_bal)
y_risk_pred_rf = rf_risk.predict(X_risk_test_scaled)
y_risk_pred_xgb = xgb_risk.predict(X_risk_test_scaled)

print("\n=== Modelo 2 - Risco ===")
print(f"RF - Acurácia: {accuracy_score(y_risk_test, y_risk_pred_rf):.4f}")
print(f"XGB - Acurácia: {accuracy_score(y_risk_test, y_risk_pred_xgb):.4f}")
best_risk = xgb_risk if f1_score(y_risk_test, y_risk_pred_xgb) >= f1_score(y_risk_test, y_risk_pred_rf) else rf_risk

# 9. Salvar artefatos (incluindo a lista de feature names)
os.makedirs('models', exist_ok=True)
joblib.dump(best_disease, 'models/modelo_doenca.pkl')
joblib.dump(scaler_disease, 'models/scaler_doenca.pkl')
joblib.dump(le_disease, 'models/label_encoder_doenca.pkl')
joblib.dump(best_risk, 'models/modelo_risco.pkl')
joblib.dump(scaler_risk, 'models/scaler_risco.pkl')
joblib.dump(feature_cols, 'models/feature_names.pkl')   # salvar nomes das features

print("\n✅ Artefatos salvos em 'models/'")