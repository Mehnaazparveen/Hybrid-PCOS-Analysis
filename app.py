"""Hybrid Multi-Stage PCOS Detection System.

Run:
    pip install -r requirements.txt
    python train_model.py          # one-time
    streamlit run app.py
"""
from pathlib import Path
import joblib
import streamlit as st
from PIL import Image
from gradcam import generate_gradcam
from preprocessing import build_feature_vector, calc_bmi
from dashboard import render_eda, render_history
from recommendations import recommend
from train_model import load_training_data

st.set_page_config(page_title="PCOS Detection System", page_icon="🩺", layout="wide")
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

/* Chart Containers */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

/* Headers */
h1, h2, h3 {
    color: #F8FAFC;
    font-weight: 700;
}

/* Plot Containers */
.stPlotlyChart {
    background-color: #111827;
    padding: 20px;
    border-radius: 20px;
    margin-bottom: 30px;
    border: 1px solid #1f2937;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.35);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#2563EB,#06B6D4);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 20px;
    font-weight: bold;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #06B6D4, #2563EB);
}

</style>
""", unsafe_allow_html=True)
MODEL_PATH = Path("models/symptom_model.joblib")

ULTRASOUND_MODEL_PATH = Path("models/pcosgen_mobilenetv2.keras")
PCOM_THRESHOLD = 0.60

if "history" not in st.session_state:
    st.session_state.history = []
if "last_risk" not in st.session_state:
    st.session_state.last_risk = None


@st.cache_resource
def load_symptom_model():
    if not MODEL_PATH.exists():
        st.warning("Symptom model not found. Run: python train_model.py")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_ultrasound_model():
    import tensorflow as tf

    if not ULTRASOUND_MODEL_PATH.exists():
        raise FileNotFoundError(
            "MobileNetV2 ultrasound model not found."
        )

    return tf.keras.models.load_model(
        str(ULTRASOUND_MODEL_PATH),
        compile=False
    )
def predict_ultrasound(image_bytes):
    import tensorflow as tf

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, (160, 160))
    image = tf.expand_dims(image, axis=0)

    model = load_ultrasound_model()

    score = float(
        model.predict(image, verbose=0)[0][0]
    )

    label = (
        "PCOM Visible"
        if score >= PCOM_THRESHOLD
        else "PCOM Not Visible"
    )

    return {
        "label": label,
        "score": score
    }

@st.cache_data
def load_dataset():
    features, target = load_training_data()
    dataset = features.copy()
    dataset["pcos"] = target
    return dataset


# ---- Sidebar nav ----
st.sidebar.title("🩺 PCOS Detection")
page = st.sidebar.radio("Navigate", ["Predict", "Dashboard", "EDA", "About"])

# ============================================================
if page == "Predict":
    st.title("Hybrid Multi-Stage PCOS Detection")
    st.caption("Step 1 — Symptom-based ML.  Step 2 — Ultrasound CNN (if needed).")

    with st.form("symptom_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 10, 70, 25)
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)
            height = st.number_input("Height (cm)", 120.0, 210.0, 160.0)
            
        with c2:
            irregular = st.radio("Irregular periods", ["No", "Yes"], horizontal=True)
            acne = st.radio("Acne", ["No", "Yes"], horizontal=True)
            hair = st.radio("Excess hair growth", ["No", "Yes"], horizontal=True)
            wgain = st.radio("Recent weight gain", ["No", "Yes"], horizontal=True)
        with c3:
            skin = st.radio("Skin darkening", ["No", "Yes"], horizontal=True)
            ff = st.radio("Frequent fast food", ["No", "Yes"], horizontal=True)
            exer = st.radio("Regular exercise", ["No", "Yes"], horizontal=True)
            
        submitted = st.form_submit_button("Predict risk", use_container_width=True)

    if submitted:
        bmi = calc_bmi(weight, height)
        user = {
            "age": age, "weight": weight, "height": height, "bmi": bmi,
            "irregular_periods": irregular, "acne": acne, "hair_growth": hair,
            "weight_gain": wgain, "skin_darkening": skin, "fast_food": ff,
            "exercise": exer,
        
        }
        model = load_symptom_model()
        if model is None:
            st.stop()
        X = build_feature_vector(user)
        prob = float(model.predict_proba(X)[0, 1])
        risk_pct = round(prob * 100, 1)
        level = "Low" if risk_pct < 40 else "Moderate" if risk_pct < 60 else "High"

        st.session_state.last_risk = {"risk_pct": risk_pct, "level": level, "user": user}
        st.session_state.history.append(
            {"age": age, "bmi": bmi, "risk_pct": risk_pct, "level": level}
        )

    # ---- Show results outside the form so we can render upload widget ----
    if st.session_state.last_risk:
        r = st.session_state.last_risk
        risk_pct, level, user = r["risk_pct"], r["level"], r["user"]
        m1, m2, m3 = st.columns(3)
        m1.metric("BMI", user["bmi"])
        m2.metric("Experimental symptom-model score", f"{risk_pct}%")
        m3.metric("Level", level)
        st.progress(min(int(risk_pct), 100))
        st.caption(
            "Low, Moderate, and High are experimental score bands, "
            "not clinically validated categories."
        )

        st.info(
            "This symptom score is an experimental ML result, "
            "not a clinically validated PCOS probability. "
            "It cannot confirm or rule out PCOS."
        )

        st.subheader("Step 2 - Ultrasound Morphology Analysis")

        st.caption(
            "Optional ultrasound analysis using MobileNetV2."
        )

        up = st.file_uploader(
            "Upload ovarian ultrasound image",
            type=["png", "jpg", "jpeg"],
            key="pcom_ultrasound"
        )

        if up is not None:
            img = Image.open(up).convert("RGB")
            image_bytes = up.getvalue()

            original_col, heatmap_col = st.columns(2)

            original_col.image(
                img,
                caption="Original Ultrasound",
                width=320
            )

            try:
                with st.spinner("Analyzing ultrasound..."):
                    res = predict_ultrasound(image_bytes)

            except Exception as error:
                st.error(
                    f"Ultrasound analysis failed: {error}"
                )

            else:
                label_col, score_col = st.columns(2)
                label_col.metric("Ultrasound classification", res["label"])
                score_col.metric("Model score", f"{res['score']:.3f}")

                try:
                    with st.spinner("Generating Grad-CAM heatmap..."):
                        heatmap, _ = generate_gradcam(
                            img,
                            load_ultrasound_model(),
                            image_bytes=image_bytes
                        )

                    heatmap_col.image(
                        heatmap,
                        caption="Grad-CAM Explanation (Experimental)",
                        width=320
                    )

                except Exception as error:
                    st.warning(f"Grad-CAM unavailable: {error}")

                st.caption(
                    "This model score is not a calibrated clinical probability. "
                    "The model predicts PCOM visibility, not PCOS diagnosis. "
                    "Grad-CAM is an experimental visualization, not a medical annotation."
                )
        st.subheader("Recommendations")
        for tip in recommend(risk_pct, user):
            st.markdown(f"- {tip}")

# ============================================================
elif page == "Dashboard":
    st.title("Dashboard")
    render_history(st.session_state.history)

# ============================================================
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    try:
        eda_dataset = load_dataset()
    except FileNotFoundError:
        st.error(
            "Clinical dataset not found. To use EDA, place "
            "`dataset/PCOS_data_without_infertility.xlsx` in the project "
            "directory. The workbook must include the `Full_new` worksheet."
        )
    else:
        render_eda(eda_dataset)

# ============================================================
else:
    st.title("About")
    st.markdown("""
**Hybrid Multi-Stage PCOS Detection System**

### Implemented models

- **Clinical symptom model:** a saved Scikit-learn Logistic Regression pipeline
  uses 11 features from `dataset/PCOS_data_without_infertility.xlsx`. The
  displayed score and Low/Moderate/High bands are experimental, not clinically
  validated.
- **Ultrasound model:** TensorFlow MobileNetV2 analyzes an uploaded ovarian
  ultrasound resized to 160 × 160 RGB. The saved model performs its own
  normalization and predicts **PCOM Visible** or **PCOM Not Visible**; it does
  not diagnose PCOS.
- **Grad-CAM:** an experimental visualization is shown beside the original
  image. It is not a medical annotation or a validated explanation.
- **Data exploration:** the EDA page summarizes the clinical symptom dataset.
  Symptom and ultrasound models were trained on separate datasets; the app
  does not combine them into a joint patient-level risk estimate.

### Evaluation and limitations

- Symptom model held-out results: accuracy **83.3%**, ROC-AUC **0.899**.
- Ultrasound screened test results on **1,395 images**: accuracy **88.10%**,
  balanced accuracy **87.97%**, ROC-AUC **0.9336**, and PCOM F1 **0.8437**.
- Source-related image overlap remains a limitation. These results are not
  clinically validated and do not establish fully independent evaluation.
- Neither model score is a calibrated clinical probability. Outputs cannot
  confirm or rule out PCOS and should not replace professional evaluation.

> Educational project — not a medical diagnosis. Always consult a clinician.
""")
