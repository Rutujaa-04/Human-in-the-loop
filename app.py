# ==============================
# 1. IMPORTS
# ==============================
import pickle
import streamlit as st
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from lime.lime_text import LimeTextExplainer
from scipy.sparse import hstack

# ==============================
# 2. LOAD SAVED MODEL FILES
# ==============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("char_vectorizer.pkl", "rb") as f:
    char_vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ==============================
# 3. TEXT CLEANING FUNCTIONS
# ==============================
def clean_resume(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

def advanced_clean(text):
    text = clean_resume(text)
    text = remove_stopwords(text)
    words = text.split()
    words = [w for w in words if 3 <= len(w) <= 15]
    return " ".join(words)

# ==============================
# 4. PREDICTION FUNCTION
# ==============================
def predict_resume(text):
    text = advanced_clean(text)

    word_vec = vectorizer.transform([text])
    char_vec = char_vectorizer.transform([text])

    vec = hstack([word_vec, char_vec])

    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)

    confidence = probs.max()
    category = le.inverse_transform([pred])[0]

    return category, confidence

# ==============================
# 5. LIME EXPLAINABILITY
# ==============================
class_names = list(le.classes_)
explainer = LimeTextExplainer(class_names=class_names)

def predict_proba_lime(texts):
    cleaned = [advanced_clean(t) for t in texts]

    word_vec = vectorizer.transform(cleaned)
    char_vec = char_vectorizer.transform(cleaned)

    vec = hstack([word_vec, char_vec])

    return model.predict_proba(vec)

def explain_prediction(text):
    exp = explainer.explain_instance(
        text,
        predict_proba_lime,
        num_features=10
    )
    return exp.as_list()

# ==============================
# 6. STREAMLIT UI
# ==============================
st.title("Human-in-the-Loop Resume Screening System")

# Input
resume = st.text_area("Paste Resume Text")

target_role = st.selectbox(
    "Select Target Role",
    list(le.classes_)
)

# ==============================
# 7. MAIN LOGIC
# ==============================
if st.button("Analyze Resume"):

    if resume.strip() == "":
        st.warning("⚠️ Please enter resume text")
    
    else:
        # Prediction
        category, confidence = predict_resume(resume)

        st.subheader("🔍 AI Prediction")
        st.write("Predicted Role:", category)
        st.write("Confidence:", round(confidence, 2))

        # Decision
        if confidence < 0.6:
            decision = "⚠️ Needs Human Review"
        elif category == target_role:
            decision = "Selected ✅"
        else:
            decision = "Rejected ❌"

        st.subheader("🤖 AI Decision")
        st.write(decision)

        # Explanation
        st.subheader("🧠 Explanation")
        explanation = explain_prediction(resume)

        for word, score in explanation:
            st.write(f"{word} : {round(score,3)}")

        # ==============================
        # HUMAN-IN-THE-LOOP
        # ==============================
        st.subheader("👤 Human Override")

        human_decision = st.selectbox(
            "Override Decision",
            ["Accept AI Decision", "Force Select", "Force Reject"]
        )

        if human_decision == "Accept AI Decision":
            final_decision = decision
        elif human_decision == "Force Select":
            final_decision = "Selected ✅"
        else:
            final_decision = "Rejected ❌"

        st.subheader("✅ Final Decision")
        st.write(final_decision)

        # ==============================
        # FEEDBACK STORAGE (BASIC)
        # ==============================
        if "feedback_data" not in st.session_state:
            st.session_state.feedback_data = []

        st.session_state.feedback_data.append({
            "resume": resume,
            "ai_prediction": decision,
            "final_decision": final_decision
        })

        st.subheader("📊 System Stats")
        st.write("Total Feedback:", len(st.session_state.feedback_data))