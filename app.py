import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image

from ui import render_ui
from utils import extract_report_items, extract_text, extract_values


st.set_page_config(page_title="AI Medical Analyzer", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #10212b;
        --muted: #5f6f78;
        --line: #d9e3e7;
        --panel: rgba(255, 255, 255, 0.86);
        --blue: #246bfd;
        --teal: #0d9488;
        --red: #dc2626;
        --green: #15803d;
        --amber: #b45309;
    }

    [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(135deg, rgba(36, 107, 253, 0.10), transparent 34%),
            linear-gradient(225deg, rgba(13, 148, 136, 0.14), transparent 36%),
            #f6fafb;
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3, p {
        font-family: "Segoe UI", Arial, sans-serif;
        letter-spacing: 0;
    }

    .hero {
        min-height: 230px;
        display: flex;
        align-items: end;
        padding: 2rem;
        margin-bottom: 1.2rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background:
            linear-gradient(90deg, rgba(16, 33, 43, 0.92), rgba(16, 33, 43, 0.60)),
            url("https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=1800&q=80");
        background-position: center;
        background-size: cover;
        box-shadow: 0 18px 48px rgba(16, 33, 43, 0.12);
    }

    .hero h1 {
        max-width: 720px;
        margin: 0.35rem 0 0.65rem;
        color: white;
        font-size: clamp(2.25rem, 4vw, 4.5rem);
        line-height: 1.02;
        font-weight: 800;
    }

    .hero p {
        max-width: 680px;
        margin: 0;
        color: rgba(255, 255, 255, 0.88);
        font-size: 1.05rem;
        line-height: 1.55;
    }

    .eyebrow {
        color: #a7f3d0;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .panel {
        min-height: 100%;
        padding: 1.25rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        box-shadow: 0 16px 40px rgba(16, 33, 43, 0.08);
    }

    .side-panel h3 {
        margin-top: 0;
        margin-bottom: 0.9rem;
        color: var(--ink);
        font-size: 1.05rem;
    }

    .check-row {
        display: grid;
        grid-template-columns: 42px 1fr;
        gap: 0.75rem;
        align-items: start;
        padding: 0.75rem 0;
        border-top: 1px solid var(--line);
    }

    .check-row span {
        width: 34px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #e6f4f1;
        color: var(--teal);
        font-weight: 800;
        font-size: 0.78rem;
    }

    .check-row p {
        margin: 0;
        color: var(--muted);
        line-height: 1.45;
    }

    .notice {
        margin-top: 1rem;
        padding: 0.95rem 1rem;
        border-left: 4px solid var(--amber);
        border-radius: 6px;
        background: #fff7ed;
        color: #7c2d12;
        font-size: 0.92rem;
    }

    .result-card {
        padding: 1.15rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: white;
        box-shadow: 0 12px 28px rgba(16, 33, 43, 0.07);
    }

    .risk-high {
        border-left: 5px solid var(--red);
    }

    .risk-low {
        border-left: 5px solid var(--green);
    }

    .small-label {
        margin: 0;
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .big-value {
        margin: 0.2rem 0 0;
        color: var(--ink);
        font-size: 1.8rem;
        font-weight: 800;
    }

    div.stButton > button {
        min-height: 46px;
        border: 0;
        border-radius: 8px;
        background: linear-gradient(90deg, var(--blue), var(--teal));
        color: white;
        font-weight: 800;
        letter-spacing: 0;
    }

    div.stButton > button:hover {
        border: 0;
        color: white;
        filter: brightness(0.98);
    }

    [data-testid="stFileUploader"] section {
        border-color: var(--line);
        border-radius: 8px;
        background: #f8fbfc;
    }

    [data-testid="stFileUploader"] button {
        border: 1px solid var(--line);
        background: white;
        color: var(--ink);
        box-shadow: none;
    }

    [data-testid="stFileUploader"] button:hover {
        border: 1px solid var(--teal);
        background: white;
        color: var(--ink);
    }

    [data-testid="stMetric"] {
        padding: 0.85rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #f8fbfc;
    }

    [data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.35rem;
    }

    [data-testid="stMetricDelta"] {
        color: var(--muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


def get_openai_client():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except st.errors.StreamlitSecretNotFoundError:
        api_key = ""

    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


model = load_model()
client = get_openai_client()


def generate_local_report(text, prediction, prob, values, report_items):
    risk = "High Risk" if int(prediction[0]) == 1 else "Low Risk"
    confidence = round(max(prob[0]) * 100, 2)
    missing = [
        label
        for label in ["Glucose", "BloodPressure", "BMI", "Age"]
        if label not in values
    ]
    detected_lines = [item["Source line"] for item in report_items[:8]]

    summary_parts = [
        f"The uploaded report was read with OCR and the diabetes-risk model returned {risk.lower()} with {confidence}% confidence.",
    ]

    if values:
        readable_values = []
        for key, value in values.items():
            label = "Blood pressure" if key == "BloodPressure" else key
            readable_values.append(f"{label}: {value:g}")
        summary_parts.append("Detected model inputs: " + ", ".join(readable_values) + ".")

    if missing:
        readable_missing = [
            "Blood pressure" if key == "BloodPressure" else key for key in missing
        ]
        summary_parts.append(
            "The report text did not clearly show "
            + ", ".join(readable_missing)
            + ", so those fields need manual confirmation."
        )

    risks = (
        "The result suggests increased diabetes-related risk. Review glucose control, weight, symptoms, family history, and any abnormal lab markers with a clinician."
        if int(prediction[0]) == 1
        else "The result suggests lower diabetes-related risk from the available information, but routine monitoring is still important if symptoms or risk factors are present."
    )

    recommendations = [
        "Confirm OCR-detected numbers against the original report before making decisions.",
        "Discuss abnormal or missing values with a qualified doctor.",
        "Repeat testing may be needed if the scan is unclear or the report is incomplete.",
    ]

    lifestyle = [
        "Prefer balanced meals with vegetables, lean protein, whole grains, and controlled sugar intake.",
        "Aim for regular physical activity if medically safe.",
        "Track sleep, stress, hydration, and follow-up tests as advised by a clinician.",
    ]

    report = [
        "### Original Doctor-Style Report",
        "",
        "**Summary**",
        " ".join(summary_parts),
        "",
        "**Detected Details From Report**",
    ]

    if detected_lines:
        report.extend([f"- {line}" for line in detected_lines])
    else:
        report.append("- No clear lab-value lines were detected. Open the Extracted text tab to check OCR quality.")

    report.extend(
        [
            "",
            "**Possible Risks**",
            f"- {risks}",
            "",
            "**Recommendations**",
            *[f"- {item}" for item in recommendations],
            "",
            "**Lifestyle Advice**",
            *[f"- {item}" for item in lifestyle],
            "",
            "**Warning**",
            "- This is an original computer-generated explanation from the uploaded report text. It is not a diagnosis or a replacement for medical care.",
        ]
    )

    return "\n".join(report)


def generate_doctor_report(text, prediction, prob, values, report_items):
    risk = "High Risk" if int(prediction[0]) == 1 else "Low Risk"
    confidence = round(max(prob[0]) * 100, 2)

    prompt = f"""
You are a professional medical AI assistant.

Analyze this medical report and respond in a short structured clinical format.

PATIENT REPORT:
{text[:1200]}

MODEL OUTPUT:
- Risk Level: {risk}
- Confidence: {confidence}%

FORMAT:
Summary:
Possible Risks:
Recommendations:
Lifestyle Advice:
Warning:

Keep it clear, practical, and professional. Do not claim this is a diagnosis.
"""

    if client is None:
        return generate_local_report(text, prediction, prob, values, report_items)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful clinical assistant that explains medical reports clearly.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=320,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return generate_local_report(text, prediction, prob, values, report_items)


def get_image_from_upload(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pages = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)
        return pages[0]

    return Image.open(uploaded_file)


def build_features(text):
    values = extract_values(text)
    defaults = {
        "Glucose": 120.0,
        "BloodPressure": 80.0,
        "BMI": 30.0,
        "Age": 30.0,
    }

    features = [
        values.get("Glucose", defaults["Glucose"]),
        values.get("BloodPressure", defaults["BloodPressure"]),
        values.get("BMI", defaults["BMI"]),
        values.get("Age", defaults["Age"]),
    ]

    return [features], values, defaults


def render_analysis_charts(prob):
    low_probability = round(prob[0][0] * 100, 2)
    high_probability = round(prob[0][1] * 100, 2)

    pie_data = pd.DataFrame(
        {
            "Risk": ["Low risk probability", "High risk probability"],
            "Probability": [low_probability, high_probability],
        }
    )

    st.caption("Risk probability split")
    fig, ax = plt.subplots(figsize=(2.7, 2.4))
    fig.patch.set_alpha(0)
    wedges, _, autotexts = ax.pie(
        pie_data["Probability"],
        labels=None,
        autopct="%1.1f%%",
        pctdistance=0.72,
        startangle=90,
        counterclock=False,
        colors=["#15803d", "#dc2626"],
        radius=0.82,
        wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 2},
        textprops={"color": "white", "fontsize": 8, "weight": "bold"},
    )
    ax.legend(
        wedges,
        pie_data["Risk"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1,
        frameon=False,
        fontsize=7,
    )
    ax.set(aspect="equal")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


if "page" not in st.session_state:
    st.session_state.page = "home"


if st.session_state.page == "home":
    analyze, uploaded_file = render_ui()

    if analyze:
        if uploaded_file is None:
            st.warning("Upload a PDF or image report first.")
            st.stop()

        st.session_state.file = uploaded_file
        st.session_state.page = "result"
        st.rerun()


elif st.session_state.page == "result":
    st.title("Medical Report Analysis")

    uploaded_file = st.session_state.file

    try:
        image = get_image_from_upload(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        st.stop()

    preview_col, analysis_col = st.columns([0.42, 0.58], gap="large")

    with preview_col:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Report preview")
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("Reading and analyzing the report..."):
        text = extract_text(image)

    if not text:
        st.error("OCR could not extract readable text from this report.")
        if st.button("Back to upload"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()

    features, values, defaults = build_features(text)
    report_items = extract_report_items(text)
    prediction = model.predict(features)
    prob = model.predict_proba(features)

    high_risk = int(prediction[0]) == 1
    risk_label = "High Risk" if high_risk else "Low Risk"
    confidence = round(max(prob[0]) * 100, 2)
    risk_class = "risk-high" if high_risk else "risk-low"

    with analysis_col:
        st.markdown(f'<div class="result-card {risk_class}">', unsafe_allow_html=True)
        st.markdown('<p class="small-label">Prediction</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="big-value">{risk_label}</p>', unsafe_allow_html=True)
        st.progress(confidence / 100)
        st.caption(f"Model confidence: {confidence}%")
        st.markdown("</div>", unsafe_allow_html=True)

        display_keys = ["Glucose", "BloodPressure", "BMI", "Age"]

        st.write("")
        pie_col, _ = st.columns([0.38, 0.62])
        with pie_col:
            render_analysis_charts(prob)

    st.divider()

    report_tab, details_tab, text_tab, values_tab = st.tabs(
        ["Doctor-style report", "Detected details", "Extracted text", "Model inputs"]
    )

    with report_tab:
        doctor_report = generate_doctor_report(text, prediction, prob, values, report_items)
        st.markdown(doctor_report)

    with details_tab:
        if report_items:
            st.dataframe(report_items, use_container_width=True, hide_index=True)
        else:
            st.info("No detailed lab-value rows were detected. Check the Extracted text tab to see what OCR read from the image.")

    with text_tab:
        st.text_area("OCR output", text, height=260)

    with values_tab:
        st.write(
            "Missing values are not shown as report values. If the model needs them, it uses internal fallback numbers only for prediction."
        )
        st.dataframe(
            [
                {
                    "Input": "Blood Pressure" if key == "BloodPressure" else key,
                    "Report value": values[key] if key in values else "Not found",
                    "Used by model": features[0][idx],
                    "Source": "OCR" if key in values else f"Internal fallback ({defaults[key]:g})",
                }
                for idx, key in enumerate(display_keys)
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.write("")
    if st.button("Back to upload"):
        st.session_state.page = "home"
        st.rerun()
