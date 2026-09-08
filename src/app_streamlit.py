import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import re

st.set_page_config(page_title="Disease Detector + Chatbot", layout="centered")

MODEL_PATH = "models/model.joblib"
LE_PATH = "models/label_encoder.joblib"
DATA_PATH = "data/dataset_updated.csv" if os.path.exists("data/dataset_updated.csv") else "data/dataset.csv"

# Load model and label encoder if available
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run train_model.py first and place model.joblib in models/")
else:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)

# Load dataset columns to get symptom list
df = pd.read_csv(DATA_PATH)
symptoms = [c for c in df.columns if c != 'disease']

st.title("Symptom-based Disease Detector + Chatbot")

st.markdown("Select symptoms you are experiencing (tick all that apply):")
cols = st.columns(2)
user_input = {}
for i, s in enumerate(symptoms):
    with cols[i%2]:
        user_input[s] = st.checkbox(s.replace('_',' ').capitalize(), key=s)

if st.button("Predict Disease"):
    X_input = np.array([int(user_input[s]) for s in symptoms]).reshape(1, -1)
    pred_idx = model.predict(X_input)[0]
    pred = le.inverse_transform([pred_idx])[0]
    st.success(f"Predicted disease: **{pred}**")
    # guidance mapping
    guidance = {
        "Common Cold": "Rest, fluids, OTC cold meds. See doctor if symptoms worsen or persist >7 days.",
        "Flu": "Rest, fluids, consider medical advice, antiviral therapy if <48 hours and high risk.",
        "COVID-19": "Isolate, test for COVID-19, seek medical care if breathing difficulty.",
        "Gastroenteritis": "Rehydrate, avoid solid food until vomiting subsides. See doctor if severe or dehydrated.",
        "Malaria": "Seek immediate medical attention - malaria can be serious.",
        "Dengue": "Seek immediate medical attention - watch for warning signs (bleeding, severe pain).",
        "Migraine": "Rest in dark room, OTC pain relievers. See neurologist for frequent migraines.",
        "Food Poisoning": "Rehydrate, see doctor if severe or prolonged.",
        "Allergy": "Antihistamines, avoid trigger, see allergist for recurrent problems.",
        "Bronchitis": "Rest, fluids, see doctor if high fever or breathing difficulty."
    }
    st.markdown("**Suggested (non-medical) guidance:**")
    st.write(guidance.get(pred, "Please consult a healthcare professional."))
    st.warning("This tool provides informative predictions only — it is NOT a substitute for professional medical advice.")

st.markdown("---")
st.header("Health Chatbot — Ask about diseases or symptoms")
st.markdown("Examples: 'What precautions for dengue?', 'How to treat fever?', 'Tell me about malaria'")

# Simple knowledge base
disease_info = {
    "dengue": {
        "summary": "Dengue is a mosquito-borne viral infection causing high fever, severe headache, pain behind the eyes, muscle and joint pain, and rash.",
        "precautions": "Avoid mosquito bites (nets, repellents), remove standing water, seek care for warning signs like bleeding or severe abdominal pain."
    },
    "malaria": {
        "summary": "Malaria is caused by Plasmodium parasites transmitted by mosquitoes. Symptoms include fever, chills, sweats, and headache.",
        "precautions": "Use mosquito nets, prophylaxis when traveling to endemic areas, seek immediate treatment if fever develops."
    },
    "covid-19": {
        "summary": "COVID-19 is a respiratory illness caused by SARS-CoV-2; symptoms vary widely from none to severe respiratory failure.",
        "precautions": "Vaccination, mask in high-risk settings, test if symptomatic, isolate when positive, seek care for breathing difficulties."
    },
    "flu": {
        "summary": "Influenza (flu) is a contagious respiratory illness causing fever, cough, sore throat, and body aches.",
        "precautions": "Annual vaccination, hand hygiene, stay home when sick, seek antivirals early if high risk."
    },
    "food poisoning": {
        "summary": "Food poisoning arises from contaminated food; symptoms include vomiting, diarrhea, and abdominal cramps.",
        "precautions": "Practice safe food handling, cook food thoroughly, stay hydrated, seek care if severe or bloody diarrhea."
    },
    "migraine": {
        "summary": "Migraine is a neurological condition with recurrent headaches, often accompanied by nausea, sensitivity to light/sound.",
        "precautions": "Identify triggers, rest in a dark quiet room, over-the-counter or prescribed meds, consult neurologist for frequent episodes."
    },
    "allergy": {
        "summary": "Allergies are immune responses to triggers like pollen, dust, or foods causing sneezing, rash, or breathing issues.",
        "precautions": "Avoid known triggers, use antihistamines, seek specialist for persistent or severe allergies."
    },
    "gastroenteritis": {
        "summary": "Gastroenteritis is inflammation of the stomach/intestines causing vomiting and diarrhea, often due to infection.",
        "precautions": "Rehydrate (oral rehydration solutions), maintain hygiene, seek medical care if severe or dehydrated."
    },
    "bronchitis": {
        "summary": "Bronchitis is inflammation of the bronchial tubes causing cough and mucus production.",
        "precautions": "Rest, fluids, avoid smoking, seek medical care if high fever or breathing difficulties."
    },
    "common cold": {
        "summary": "A mild viral upper respiratory infection with runny nose, sore throat, and cough.",
        "precautions": "Rest, fluids, humidifier, OTC symptom relief."
    }
}

# Chat handling
user_q = st.text_input("Type your question here")
if st.button("Ask"):
    q = user_q.strip().lower()
    if not q:
        st.info("Please type a question about a disease or symptom.")
    else:
        answered = False
        # check for disease keywords
        for d in disease_info.keys():
            if d in q:
                info = disease_info[d]
                st.subheader(d.capitalize())
                st.write(info["summary"])
                st.markdown("**Precautions:**")
                st.write(info["precautions"])
                answered = True
                break
        # check for symptom keywords
        if not answered:
            for s in symptoms:
                name = s.replace('_',' ')
                if name in q:
                    st.subheader(name.capitalize())
                    st.write("Symptom noted in dataset. Select it in the predictor to check possible diseases. If severe, seek medical help.")
                    answered = True
                    break
        if not answered:
            # fallback: simple keyword responses
            if re.search(r"precaution|prevent|avoid|how to prevent", q):
                st.write("General precautions: hand hygiene, masks in crowded places, avoid mosquito bites, safe food handling, vaccination where available.")
            else:
                st.write("Sorry, I couldn't find a direct answer. Try mentioning a disease name (e.g., dengue) or symptom (e.g., fever).")

st.caption("Note: Chatbot provides general information only and is not a replacement for professional medical advice.")
