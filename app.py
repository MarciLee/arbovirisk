# app.py
import streamlit as st
import numpy as np
import joblib

# Carregar modelos e artefatos
modelo_doenca = joblib.load('models/modelo_doenca.pkl')
scaler_doenca = joblib.load('models/scaler_doenca.pkl')
le_doenca = joblib.load('models/label_encoder_doenca.pkl')
modelo_risco = joblib.load('models/modelo_risco.pkl')
scaler_risco = joblib.load('models/scaler_risco.pkl')
feature_names = joblib.load('models/feature_names.pkl')  # lista de 68 sintomas

st.set_page_config(page_title="ArboviRisk", page_icon="🦟")
st.title("🦟 ArboviRisk")
st.markdown("Selecione os sintomas do paciente para obter um diagnóstico provável.")

# Criar checkboxes para cada sintoma
cols = st.columns(3)
inputs = {}
for i, sintoma in enumerate(feature_names):
    with cols[i % 3]:
        inputs[sintoma] = st.checkbox(sintoma)

if st.button("🔍 Diagnosticar", type="primary"):
    # Montar vetor na ordem correta
    features = np.array([[1 if inputs[s] else 0 for s in feature_names]])
    # Normalizar
    features_scaled = scaler_doenca.transform(features)
    # Predição da doença
    pred = modelo_doenca.predict(features_scaled)[0]
    probas = modelo_doenca.predict_proba(features_scaled)[0]
    doenca = le_doenca.inverse_transform([pred])[0]
    confianca = max(probas)
    
    st.subheader("📊 Resultado do Diagnóstico")
    st.write(f"**Doença provável:** {doenca}")
    st.write(f"**Confiança:** {confianca:.2%}")
    
    # Se for dengue, classificar risco
    if doenca == "Dengue":
        risco_scaled = scaler_risco.transform(features)
        risco_pred = modelo_risco.predict(risco_scaled)[0]
        risco_proba = modelo_risco.predict_proba(risco_scaled)[0][1]
        risco_label = "Dengue com sinais de alarme (risco)" if risco_pred == 1 else "Dengue clássica"
        st.write(f"**Classificação de risco:** {risco_label}")
        st.write(f"**Confiança do risco:** {risco_proba:.2%}")
    
    st.info("⚠️ Esta ferramenta é apenas um auxílio ao diagnóstico. Procure um médico em caso de sintomas graves.")