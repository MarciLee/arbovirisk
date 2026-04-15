# app.py v 2.2.0
import streamlit as st
import numpy as np
import joblib

@st.cache_resource
def load_models():
    modelo_doenca = joblib.load('models/modelo_doenca.pkl')
    scaler_doenca = joblib.load('models/scaler_doenca.pkl')
    le_doenca = joblib.load('models/label_encoder_doenca.pkl')
    modelo_risco = joblib.load('models/modelo_risco.pkl')
    scaler_risco = joblib.load('models/scaler_risco.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return (modelo_doenca, scaler_doenca, le_doenca,
            modelo_risco, scaler_risco, feature_names)

try:
    (modelo_doenca, scaler_doenca, le_doenca,
     modelo_risco, scaler_risco, feature_names) = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"Erro ao carregar modelos: {e}")
    model_loaded = False

sintomas_pt = {
    'Fever': 'Febre',
    'Headache': 'Dor de cabeça',
    'Myalgia': 'Dor muscular',
    'Arthralgia': 'Dor nas articulações',
    'Rash': 'Manchas na pele',
    'Retro-orbital pain': 'Dor atrás dos olhos',
    'Vomiting': 'Vômito',
    'Abdominal pain': 'Dor abdominal',
    'Conjunctivitis': 'Conjuntivite',
    'Pruritus': 'Coceira',
    'Lymphadenopathy': 'Ínguas',
    'Leucopenia': 'Queda de leucócitos',
    'Thrombocytopenia': 'Queda de plaquetas'
}

def get_sintoma_nome(codigo):
    return sintomas_pt.get(codigo, codigo)

st.set_page_config(page_title="ArboviRisk", page_icon="🦟", layout="wide")
st.title("🦟 ArboviRisk - Diagnóstico Inteligente")
st.markdown("Selecione os sintomas do paciente para obter um diagnóstico provável entre **Dengue, Zika e Chikungunya**.")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865918.png", width=100)
    st.markdown("## Sobre")
    st.info(
        "**Acurácia do modelo:** ~98% (Random Forest)\n\n"
        "**Base:** 48.214 casos sintéticos (Zenodo)\n\n"
        "**Diretrizes:** PAHO 2022\n\n"
        "**Doenças consideradas:** Dengue, Zika, Chikungunya"
    )
    st.markdown("---")
    st.markdown("### Recomendações")
    st.caption("🩺 **Dengue clássica:** hidratação e repouso.")
    st.caption("🚨 **Dengue com sinais de alarme:** procure atendimento urgente.")
    st.caption("🦟 **Zika/Chikungunya:** alívio dos sintomas e acompanhamento médico.")

cols = st.columns(3)
inputs = {}
for i, sintoma_codigo in enumerate(feature_names):
    rotulo = get_sintoma_nome(sintoma_codigo)
    with cols[i % 3]:
        inputs[sintoma_codigo] = st.checkbox(rotulo)

THRESHOLD = 0.75

if st.button("🔍 Diagnosticar", type="primary"):
    if not model_loaded:
        st.error("Modelos não carregados.")
    else:
        features = np.array([[1 if inputs[s] else 0 for s in feature_names]])
        features_scaled = scaler_doenca.transform(features)
        probas = modelo_doenca.predict_proba(features_scaled)[0]
        pred = np.argmax(probas)
        confianca = max(probas)
        
        st.subheader("📊 Resultado do Diagnóstico")
        
        if confianca < THRESHOLD:
            st.warning("**Diagnóstico:** Não identificado")
            st.write(f"Confiança máxima: {confianca:.0%} (abaixo do limiar de {THRESHOLD:.0%})")
            st.info("Os sintomas não se enquadram claramente nas doenças monitoradas. Consulte um médico.")
        else:
            doenca = le_doenca.inverse_transform([pred])[0]
            st.write(f"**Doença provável:** {doenca}")
            st.write(f"**Confiança:** {confianca:.0%}")
            st.progress(confianca)
            
            if doenca == "Dengue":
                risco_scaled = scaler_risco.transform(features)
                risco_pred = modelo_risco.predict(risco_scaled)[0]
                risco_proba = modelo_risco.predict_proba(risco_scaled)[0][1]
                risco_label = "Dengue com sinais de alarme (risco)" if risco_pred == 1 else "Dengue clássica"
                st.write(f"**Classificação de risco:** {risco_label}")
                st.write(f"**Confiança do risco:** {risco_proba:.0%}")
                st.progress(risco_proba)
                if risco_pred == 1:
                    st.error("🚨 Recomendação: RISCO! Procure atendimento médico imediatamente.")
                else:
                    st.success("🩺 Recomendação: Hidratação, repouso e monitoramento.")
            elif doenca == "Zika":
                st.info("🦟 Recomendação: Repouso, hidratação e evite AAS. Gestantes: acompanhamento especial.")
            elif doenca == "Chikungunya":
                st.warning("🦟 Recomendação: Analgésicos para dores articulares e repouso. Dores podem persistir.")
        
        st.info("⚠️ Ferramenta auxiliar. Em caso de sintomas graves, procure um médico.")