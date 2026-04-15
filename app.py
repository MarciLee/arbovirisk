# app.py
import streamlit as st
import numpy as np
import joblib

# Carregar modelos e artefatos (com cache)
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

# Mapeamento de códigos para nomes de sintomas em português (apenas os selecionados)
sintomas_pt = {
    'Fever': 'Febre',
    'Headache': 'Dor de cabeça',
    'Myalgia': 'Dor muscular',
    'Arthralgia': 'Dor nas articulações',
    'Rash': 'Manchas na pele',
    'Retro-orbital pain': 'Dor atrás dos olhos',
    'Vomiting': 'Vômito',
    'Abdominal pain': 'Dor abdominal',
    'Diarrhea': 'Diarreia',
    'Cough': 'Tosse',
    'Fatigue': 'Fadiga',
    'Pruritus': 'Coceira',
    'Conjunctivitis': 'Conjuntivite',
    'Lymphadenopathy': 'Ínguas (linfonodos inchados)',
    'Petechiae': 'Petéquias (pequenas manchas vermelhas)',
    'Bleeding': 'Sangramento',
    'Mucosal bleeding': 'Sangramento em mucosas',
    'Lethargy': 'Letargia (sonolência excessiva)',
    'Hepatomegaly': 'Aumento do fígado',
    'Plasma leakage': 'Extravasamento de plasma',
    'Shock': 'Choque circulatório',
    'Impaired consciousness': 'Alteração da consciência',
    'Leucopenia': 'Queda de leucócitos',
    'Thrombocytopenia': 'Queda de plaquetas'
}

def get_sintoma_nome(codigo):
    return sintomas_pt.get(codigo, codigo)

# Configuração da página
st.set_page_config(page_title="ArboviRisk", page_icon="🦟", layout="wide")
st.title("🦟 ArboviRisk - Diagnóstico Inteligente")
st.markdown("Selecione os sintomas do paciente para obter um diagnóstico provável entre **Dengue, Zika e Chikungunya**.")

# Sidebar com informações
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865918.png", width=100)
    st.markdown("## Sobre")
    st.info(
        "**Acurácia do modelo:** 98,2% (Random Forest)\n\n"
        "**Base:** 48.214 casos sintéticos (Zenodo)\n\n"
        "**Diretrizes:** PAHO 2022\n\n"
        "**Doenças consideradas:** Dengue, Zika, Chikungunya"
    )
    st.markdown("---")
    st.markdown("### Recomendações")
    st.caption("🩺 **Dengue clássica:** hidratação e repouso.")
    st.caption("🚨 **Dengue com sinais de alarme:** procure atendimento urgente.")
    st.caption("🦟 **Zika/Chikungunya:** alívio dos sintomas e acompanhamento médico.")

# Criar checkboxes para cada sintoma (apenas os relevantes)
cols = st.columns(3)
inputs = {}
for i, sintoma_codigo in enumerate(feature_names):
    rotulo = get_sintoma_nome(sintoma_codigo)
    with cols[i % 3]:
        inputs[sintoma_codigo] = st.checkbox(rotulo)

# Limiar de confiança para "Não identificado"
THRESHOLD = 0.6

if st.button("🔍 Diagnosticar", type="primary"):
    if not model_loaded:
        st.error("Modelos não carregados. Verifique os arquivos na pasta 'models'.")
    else:
        # Montar vetor na ordem correta
        features = np.array([[1 if inputs[s] else 0 for s in feature_names]])
        # Normalizar
        features_scaled = scaler_doenca.transform(features)
        # Predição da doença
        probas = modelo_doenca.predict_proba(features_scaled)[0]
        pred = np.argmax(probas)
        confianca = max(probas)
        
        st.subheader("📊 Resultado do Diagnóstico")
        
        if confianca < THRESHOLD:
            doenca = "Não identificado"
            st.warning(f"**Diagnóstico:** {doenca}")
            st.write(f"**Confiança máxima:** {confianca:.0%} (abaixo do limiar de {THRESHOLD:.0%})")
            st.info("Os sintomas informados não se enquadram claramente em nenhuma das doenças monitoradas. Consulte um médico para avaliação clínica.")
        else:
            doenca = le_doenca.inverse_transform([pred])[0]
            st.write(f"**Doença provável:** {doenca}")
            st.write(f"**Confiança:** {confianca:.0%}")
            st.progress(confianca)
            
            # Se for dengue, classificar risco
            if doenca == "Dengue":
                risco_scaled = scaler_risco.transform(features)
                risco_pred = modelo_risco.predict(risco_scaled)[0]
                risco_proba = modelo_risco.predict_proba(risco_scaled)[0][1]
                risco_label = "Dengue com sinais de alarme (risco)" if risco_pred == 1 else "Dengue clássica"
                st.write(f"**Classificação de risco:** {risco_label}")
                st.write(f"**Confiança do risco:** {risco_proba:.0%}")
                st.progress(risco_proba)
                
                if risco_pred == 1:
                    st.error("🚨 **Recomendação:** RISCO! Procure atendimento médico imediatamente.")
                else:
                    st.success("🩺 **Recomendação:** Hidratação, repouso e monitoramento. Procure um posto de saúde se os sintomas piorarem.")
            elif doenca == "Zika":
                st.info("🦟 **Recomendação:** Repouso, hidratação e evite medicamentos à base de ácido acetilsalicílico.")
            elif doenca == "Chikungunya":
                st.warning("🦟 **Recomendação:** Analgésicos para dores articulares e repouso.")
        
        st.info("⚠️ Esta ferramenta é apenas um auxílio ao diagnóstico. Procure um médico em caso de sintomas graves.")