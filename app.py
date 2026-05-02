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

    .analysis-loader {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 1rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.86);
        box-shadow: 0 12px 30px rgba(16, 33, 43, 0.07);
    }

    .loader-pulse {
        width: 42px;
        height: 42px;
        position: relative;
        border-radius: 999px;
        background: #e6f4f1;
    }

    .loader-pulse::before,
    .loader-pulse::after {
        content: "";
        position: absolute;
        inset: 11px;
        border-radius: 999px;
        background: var(--teal);
        animation: pulse-ring 1.35s ease-in-out infinite;
    }

    .loader-pulse::after {
        inset: 17px 8px;
        border-radius: 999px;
        background: white;
        box-shadow: 0 -9px 0 var(--teal), 0 9px 0 var(--teal);
        animation: heartbeat 1.35s ease-in-out infinite;
    }

    .loader-copy strong {
        display: block;
        color: var(--ink);
        font-size: 0.95rem;
    }

    .loader-copy span {
        color: var(--muted);
        font-size: 0.86rem;
    }

    @keyframes pulse-ring {
        0%, 100% { transform: scale(0.88); opacity: 0.75; }
        50% { transform: scale(1.08); opacity: 1; }
    }

    @keyframes heartbeat {
        0%, 100% { transform: scaleX(0.92); }
        50% { transform: scaleX(1.08); }
    }

    @keyframes ai-dots {
        0% { content: ""; }
        25% { content: "."; }
        50% { content: ".."; }
        75%, 100% { content: "..."; }
    }

    .st-key-doctor_ai_float {
        position: fixed;
        right: 24px;
        bottom: 24px;
        width: min(380px, calc(100vw - 32px));
        max-height: 76vh;
        overflow-y: auto;
        z-index: 9999;
        padding: 0.85rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 18px 48px rgba(16, 33, 43, 0.18);
    }

    .st-key-doctor_ai_float:has(#doctor-ai-closed) {
        width: 188px;
        padding: 0;
        border: 0;
        background: transparent;
        box-shadow: none;
        overflow: visible;
    }

    .st-key-doctor_ai_float:has(#doctor-ai-closed) button {
        width: 58px;
        height: 58px;
        min-height: 58px;
        padding: 0;
        border-radius: 999px;
        background: linear-gradient(135deg, #e11d48, #f97316);
        color: white;
        box-shadow: 0 14px 34px rgba(225, 29, 72, 0.34);
        font-size: 0.95rem;
    }

    .st-key-doctor_ai_float:has(#doctor-ai-closed) .stButton {
        display: flex;
        justify-content: flex-end;
    }

    .doctor-ai-label {
        width: fit-content;
        margin: 0 0 0.45rem auto;
        padding: 0.5rem 0.7rem;
        border: 1px solid rgba(225, 29, 72, 0.18);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.95);
        color: var(--ink);
        box-shadow: 0 10px 28px rgba(16, 33, 43, 0.12);
        font-size: 0.84rem;
        font-weight: 800;
        white-space: nowrap;
    }

    .doctor-ai-dots::after {
        content: "";
        display: inline-block;
        width: 1.1em;
        text-align: left;
        animation: ai-dots 1.2s steps(4, end) infinite;
    }

    .doctor-chat-title {
        margin: 0 0 0.25rem;
        color: var(--ink);
        font-size: 1rem;
        font-weight: 800;
    }

    .doctor-chat-note {
        margin: 0 0 0.75rem;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.4;
    }

    .st-key-doctor_ai_float [data-testid="stChatMessage"] {
        padding: 0.45rem 0.6rem;
        border-radius: 8px;
        background: #f8fbfc;
        color: var(--ink);
    }

    .st-key-doctor_ai_float [data-testid="stChatMessage"] p,
    .st-key-doctor_ai_float label,
    .st-key-doctor_ai_float p,
    .st-key-doctor_ai_float span {
        color: var(--ink);
    }

    .st-key-doctor_ai_float textarea {
        min-height: 78px;
        border: 1px solid var(--line);
        background: white;
        color: var(--ink);
        caret-color: var(--ink);
    }

    .st-key-doctor_ai_float textarea::placeholder {
        color: var(--muted);
        opacity: 1;
    }

    .st-key-doctor_ai_float [data-testid="stFormSubmitButton"] button {
        border: 1px solid var(--line);
        background: white;
        color: var(--ink);
        box-shadow: none;
    }

    .st-key-doctor_ai_float [data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--teal);
        background: #f8fbfc;
        color: var(--ink);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
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
REQUIRED_INPUTS = ["Glucose", "BloodPressure", "BMI", "Age"]
MODEL_MEAN_VALUES = {
    "Glucose": 120.0,
    "BloodPressure": 80.0,
    "BMI": 30.0,
    "Age": 30.0,
}


def readable_input_name(key):
    return "Blood pressure" if key == "BloodPressure" else key


def format_model_inputs(model_values, imputed_inputs):
    return ", ".join(
        [
            f"{readable_input_name(key)}: {model_values[key]:g}"
            + (" (mean filled)" if key in imputed_inputs else " (read from report)")
            for key in REQUIRED_INPUTS
        ]
    )


def generate_local_report(text, prediction, prob, values, model_values, imputed_inputs, report_items):
    risk = "High Risk" if int(prediction[0]) == 1 else "Low Risk"
    confidence = round(max(prob[0]) * 100, 2)
    detected_lines = [item["Source line"] for item in report_items[:8]]

    summary_parts = [
        f"The uploaded report was read with OCR and the diabetes-risk model returned {risk.lower()} with {confidence}% confidence.",
        "Model inputs used: " + format_model_inputs(model_values, imputed_inputs) + ".",
    ]

    if values:
        readable_values = []
        for key, value in values.items():
            label = "Blood pressure" if key == "BloodPressure" else key
            readable_values.append(f"{label}: {value:g}")
        summary_parts.append("Detected model inputs: " + ", ".join(readable_values) + ".")

    if imputed_inputs:
        readable_missing = [
            readable_input_name(key) for key in imputed_inputs
        ]
        summary_parts.append(
            "The report text did not clearly show "
            + ", ".join(readable_missing)
            + ", so model mean values were used for those fields."
        )

    risks = (
        "The result suggests increased diabetes-related risk. Review glucose control, weight, symptoms, family history, and any abnormal lab markers with a clinician."
        if int(prediction[0]) == 1
        else "The result suggests lower diabetes-related risk from the available information, but routine monitoring is still important if symptoms or risk factors are present."
    )

    recommendations = [
        "Confirm OCR-detected and mean-filled numbers against the original report before making decisions.",
        "Discuss abnormal or mean-filled values with a qualified doctor.",
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
        report.append("- No clear lab-value lines were detected. The scan may be unclear or the values may use an unsupported format.")

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


def generate_incomplete_report(values, missing_inputs, report_items):
    detected_values = ", ".join(
        [
            f"{readable_input_name(key)}: {value:g}"
            for key, value in values.items()
        ]
    )
    detected_values = detected_values or "No required model values were found."
    missing_values = ", ".join([readable_input_name(key) for key in missing_inputs])
    detected_lines = [item["Source line"] for item in report_items[:8]]

    report = [
        "### Doctor-Style Report",
        "",
        "**Summary**",
        "The uploaded report was read with OCR, but the app could not make a diabetes-risk prediction because required model values are missing.",
        "",
        "**Detected Values**",
        f"- {detected_values}",
        "",
        "**Missing Values Needed For Prediction**",
        f"- {missing_values}",
        "",
        "**Detected Details From Report**",
    ]

    if detected_lines:
        report.extend([f"- {line}" for line in detected_lines])
    else:
        report.append("- No clear lab-value lines were detected. The scan may be unclear or the values may use an unsupported format.")

    report.extend(
        [
            "",
            "**Recommendation**",
            "- Upload a clearer report or make sure the report includes glucose, blood pressure, BMI, and age before prediction.",
            "- Confirm all OCR-detected values against the original report.",
            "",
            "**Warning**",
            "- No diagnosis or risk score was generated because the required values were incomplete.",
        ]
    )

    return "\n".join(report)


def generate_doctor_report(text, prediction, prob, values, model_values, imputed_inputs, report_items):
    risk = "High Risk" if int(prediction[0]) == 1 else "Low Risk"
    confidence = round(max(prob[0]) * 100, 2)
    detected_values = ", ".join(
        [
            f"{'Blood pressure' if key == 'BloodPressure' else key}: {value:g}"
            for key, value in values.items()
        ]
    )
    detected_values = detected_values or "No clear model input values were found."
    model_input_summary = format_model_inputs(model_values, imputed_inputs)
    imputed_summary = ", ".join(
        [readable_input_name(key) for key in imputed_inputs]
    )
    imputed_summary = imputed_summary or "None"
    detected_lines = "\n".join(
        [
            f"- {item['Test']}: {item['Value']} {item['Unit']} ({item['Status']})"
            for item in report_items[:10]
        ]
    )
    detected_lines = detected_lines or "- No detailed lab-value rows were detected."

    prompt = f"""
You are a professional medical AI assistant.

Analyze this medical report and respond in a short structured clinical format.

PATIENT REPORT:
{text[:1200]}

DETECTED VALUES:
- Values read from report: {detected_values}
- Model inputs used for prediction: {model_input_summary}
- Mean-filled values: {imputed_summary}
{detected_lines}

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
        return generate_local_report(text, prediction, prob, values, model_values, imputed_inputs, report_items)

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
        return generate_local_report(text, prediction, prob, values, model_values, imputed_inputs, report_items)


def build_chat_context(text, prediction, prob, values, model_values, imputed_inputs, report_items):
    risk = "High Risk" if int(prediction[0]) == 1 else "Low Risk"
    confidence = round(max(prob[0]) * 100, 2)
    detected_values = ", ".join(
        [
            f"{'Blood pressure' if key == 'BloodPressure' else key}: {value:g}"
            for key, value in values.items()
        ]
    )
    detected_values = detected_values or "No clear model input values were found."
    model_input_summary = format_model_inputs(model_values, imputed_inputs)
    imputed_summary = ", ".join(
        [readable_input_name(key) for key in imputed_inputs]
    )
    imputed_summary = imputed_summary or "None"
    details = "\n".join(
        [
            f"- {item['Test']}: {item['Value']} {item['Unit']} ({item['Status']}) | {item['Source line']}"
            for item in report_items[:12]
        ]
    )
    details = details or "- No detailed lab-value rows were detected."

    return f"""
Uploaded medical report context:
- Diabetes risk model result: {risk}
- Model confidence: {confidence}%
- Values read from report: {detected_values}
- Model inputs used for prediction: {model_input_summary}
- Mean-filled values: {imputed_summary}
- Explain clearly when a prediction used mean-filled values instead of report-read values.

Detected report lines:
{details}

OCR text excerpt:
{text[:1500]}
"""


def generate_local_chat_reply(question, prediction, prob, values, model_values, imputed_inputs, report_items):
    risk = "high" if int(prediction[0]) == 1 else "lower"
    confidence = round(max(prob[0]) * 100, 2)
    detected_values = ", ".join(
        [
            f"{'blood pressure' if key == 'BloodPressure' else key.lower()}: {value:g}"
            for key, value in values.items()
        ]
    )

    reply = [
        f"Based on the uploaded report, the model suggests {risk} diabetes-related risk with {confidence}% confidence.",
        "Model inputs used: " + format_model_inputs(model_values, imputed_inputs) + ".",
    ]

    if detected_values:
        reply.append(f"I found these usable values: {detected_values}.")
    elif report_items:
        detected_tests = ", ".join(
            [
                f"{item['Test']}: {item['Value']} {item['Unit']}".strip()
                for item in report_items[:6]
            ]
        )
        reply.append(f"I found these report values: {detected_tests}.")
    else:
        reply.append("I could not clearly find the main model values in the report text, so please verify the scan or enter values manually.")

    reply.extend(
        [
            "Mean-filled values reduce reliability, so treat the score as an estimate.",
            "For your question, use this as guidance only: confirm the report with a qualified doctor, especially if glucose is high, symptoms are present, or the OCR text looks incorrect.",
            "This chat cannot diagnose, prescribe medicine, or replace medical care.",
        ]
    )

    return " ".join(reply)


def generate_chat_reply(question, chat_messages, text, prediction, prob, values, model_values, imputed_inputs, report_items):
    if client is None:
        return generate_local_chat_reply(question, prediction, prob, values, model_values, imputed_inputs, report_items)

    context = build_chat_context(text, prediction, prob, values, model_values, imputed_inputs, report_items)
    recent_messages = chat_messages[-8:]
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI medical assistant inside a medical report analyzer. "
                "Explain reports in simple language, ask users to seek urgent care for severe symptoms, "
                "and never claim to diagnose, prescribe, or replace a licensed doctor. "
                "Use the provided report context when relevant. Keep answers concise and practical."
            ),
        },
        {"role": "system", "content": context},
        *recent_messages,
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=360,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        return generate_local_chat_reply(question, prediction, prob, values, model_values, imputed_inputs, report_items)


def generate_incomplete_chat_reply(question, chat_messages, text, values, missing_inputs, report_items):
    detected_values = ", ".join(
        [
            f"{readable_input_name(key)}: {value:g}"
            for key, value in values.items()
        ]
    )
    detected_values = detected_values or "No required model values were found."
    missing_values = ", ".join([readable_input_name(key) for key in missing_inputs])
    details = "\n".join(
        [
            f"- {item['Test']}: {item['Value']} {item['Unit']} | {item['Source line']}"
            for item in report_items[:12]
        ]
    )
    details = details or "- No detailed lab-value rows were detected."

    if client is None:
        return (
            "I cannot calculate a diabetes-risk prediction from this report because "
            f"these required values are missing: {missing_values}. "
            f"Detected values: {detected_values}. Upload a clearer report or confirm the missing values with a doctor."
        )

    recent_messages = chat_messages[-8:]
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI medical assistant inside a medical report analyzer. "
                "The app did not generate a diabetes-risk prediction because required model inputs are missing. "
                "Do not mention a risk score or confidence. Explain the available report text and tell the user what values are missing. "
                "Never diagnose, prescribe, or replace a licensed doctor."
            ),
        },
        {
            "role": "system",
            "content": f"""
Detected required values: {detected_values}
Missing required values: {missing_values}

Detected report lines:
{details}

OCR text excerpt:
{text[:1500]}
""",
        },
        *recent_messages,
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=320,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "I cannot calculate a diabetes-risk prediction because the required report values are incomplete. "
            f"Missing: {missing_values}. Please verify the report scan or consult a qualified doctor."
        )


def render_doctor_chat(text, prediction, prob, values, model_values, imputed_inputs, report_items):
    if "doctor_chat_open" not in st.session_state:
        st.session_state.doctor_chat_open = False

    if not st.session_state.doctor_chat_open:
        st.markdown(
            '<span id="doctor-ai-closed"></span><div class="doctor-ai-label">AI Doctor<span class="doctor-ai-dots"></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Dr AI", key="open_doctor_ai_chat", help="Open Doctor AI chat"):
            st.session_state.doctor_chat_open = True
            st.rerun()
        return

    st.markdown('<span id="doctor-ai-open"></span>', unsafe_allow_html=True)
    top_left, top_right = st.columns([0.78, 0.22])
    with top_left:
        st.markdown(
            """
            <p class="doctor-chat-title">Doctor AI Chat Bot</p>
            <p class="doctor-chat-note">Ask about this report. This AI can explain results, but it cannot diagnose or prescribe.</p>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("Close", key="close_doctor_ai_chat"):
            st.session_state.doctor_chat_open = False
            st.rerun()

    if "doctor_chat_messages" not in st.session_state:
        st.session_state.doctor_chat_messages = [
            {
                "role": "assistant",
                "content": "Hi, I can help explain this report and suggest what to discuss with a doctor. What would you like to know?",
            }
        ]

    chat_box = st.container()
    with chat_box:
        for message in st.session_state.doctor_chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    with st.form("doctor_ai_chat_form", clear_on_submit=True):
        question = st.text_area(
            "Ask the Doctor AI Chat Bot",
            placeholder="Example: What does my glucose result mean?",
            height=90,
        )
        send = st.form_submit_button("Send question", use_container_width=True)

    if send and not question.strip():
        st.warning("Type a question for the Doctor AI Chat Bot first.")
        return

    if send:
        question = question.strip()
        st.session_state.doctor_chat_messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            thinking = st.empty()
            thinking.markdown(
                """
                <div class="analysis-loader">
                    <div class="loader-pulse"></div>
                    <div class="loader-copy">
                        <strong>Doctor AI is reviewing the report</strong>
                        <span>Checking the report context before answering.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            reply = generate_chat_reply(
                question,
                st.session_state.doctor_chat_messages,
                text,
                prediction,
                prob,
                values,
                model_values,
                imputed_inputs,
                report_items,
            )
            thinking.empty()
            st.markdown(reply)

        st.session_state.doctor_chat_messages.append(
            {"role": "assistant", "content": reply}
        )
        st.rerun()


def render_incomplete_doctor_chat(text, values, missing_inputs, report_items):
    if "doctor_chat_open" not in st.session_state:
        st.session_state.doctor_chat_open = False

    if not st.session_state.doctor_chat_open:
        st.markdown(
            '<span id="doctor-ai-closed"></span><div class="doctor-ai-label">AI Doctor<span class="doctor-ai-dots"></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Dr AI", key="open_doctor_ai_chat_incomplete", help="Open Doctor AI chat"):
            st.session_state.doctor_chat_open = True
            st.rerun()
        return

    st.markdown('<span id="doctor-ai-open"></span>', unsafe_allow_html=True)
    top_left, top_right = st.columns([0.78, 0.22])
    with top_left:
        st.markdown(
            """
            <p class="doctor-chat-title">Doctor AI Chat Bot</p>
            <p class="doctor-chat-note">Ask about this report. No prediction was generated because required values are missing.</p>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("Close", key="close_doctor_ai_chat_incomplete"):
            st.session_state.doctor_chat_open = False
            st.rerun()

    if "doctor_chat_messages" not in st.session_state:
        st.session_state.doctor_chat_messages = [
            {
                "role": "assistant",
                "content": "I can explain what was detected, but I cannot calculate risk until glucose, blood pressure, BMI, and age are available.",
            }
        ]

    for message in st.session_state.doctor_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("doctor_ai_incomplete_chat_form", clear_on_submit=True):
        question = st.text_area(
            "Ask the Doctor AI Chat Bot",
            placeholder="Example: Which values are missing from this report?",
            height=90,
        )
        send = st.form_submit_button("Send question", use_container_width=True)

    if send and not question.strip():
        st.warning("Type a question for the Doctor AI Chat Bot first.")
        return

    if send:
        question = question.strip()
        st.session_state.doctor_chat_messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            thinking = st.empty()
            thinking.markdown(
                """
                <div class="analysis-loader">
                    <div class="loader-pulse"></div>
                    <div class="loader-copy">
                        <strong>Doctor AI is reviewing the report</strong>
                        <span>Checking the report context before answering.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            reply = generate_incomplete_chat_reply(
                question,
                st.session_state.doctor_chat_messages,
                text,
                values,
                missing_inputs,
                report_items,
            )
            thinking.empty()
            st.markdown(reply)

        st.session_state.doctor_chat_messages.append(
            {"role": "assistant", "content": reply}
        )
        st.rerun()


def get_image_from_upload(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pages = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)
        return pages[0]

    return Image.open(uploaded_file)


def build_features(text):
    values = extract_values(text)
    imputed_inputs = [key for key in REQUIRED_INPUTS if key not in values]
    model_values = {
        key: values.get(key, MODEL_MEAN_VALUES[key]) for key in REQUIRED_INPUTS
    }
    features = [model_values[key] for key in REQUIRED_INPUTS]

    return [features], values, model_values, imputed_inputs


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
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    fig.patch.set_alpha(0)
    wedges, _, autotexts = ax.pie(
        pie_data["Probability"],
        labels=None,
        autopct="%1.1f%%",
        pctdistance=0.72,
        startangle=90,
        counterclock=False,
        colors=["#15803d", "#dc2626"],
        radius=0.9,
        wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 2},
        textprops={"color": "white", "fontsize": 9, "weight": "bold"},
    )
    ax.legend(
        wedges,
        pie_data["Risk"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1,
        frameon=False,
        fontsize=8,
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
        st.session_state.doctor_chat_messages = [
            {
                "role": "assistant",
                "content": "Hi, I can help explain this report and suggest what to discuss with a doctor. What would you like to know?",
            }
        ]
        st.session_state.doctor_chat_open = False
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

    loader = st.empty()
    loader.markdown(
        """
        <div class="analysis-loader">
            <div class="loader-pulse"></div>
            <div class="loader-copy">
                <strong>Scanning your medical report</strong>
                <span>Reading text, finding values, and preparing the Doctor AI chat.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    text = extract_text(image)
    loader.empty()

    if not text:
        st.error("OCR could not extract readable text from this report.")
        if st.button("Back to upload"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()

    features, values, model_values, imputed_inputs = build_features(text)
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
        if imputed_inputs:
            filled_labels = ", ".join(
                [readable_input_name(key) for key in imputed_inputs]
            )
            st.caption(f"Mean values used for missing fields: {filled_labels}.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        pie_col, _ = st.columns([0.44, 0.56])
        with pie_col:
            render_analysis_charts(prob)

    st.divider()

    st.subheader("Doctor-style report")
    doctor_report = generate_doctor_report(
        text,
        prediction,
        prob,
        values,
        model_values,
        imputed_inputs,
        report_items,
    )
    st.markdown(doctor_report)

    with st.container(key="doctor_ai_float"):
        render_doctor_chat(
            text,
            prediction,
            prob,
            values,
            model_values,
            imputed_inputs,
            report_items,
        )

    st.write("")
    if st.button("Back to upload"):
        st.session_state.page = "home"
        st.rerun()
