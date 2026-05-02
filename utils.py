import streamlit as st
import pytesseract
import cv2
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------- OCR FUNCTION ----------------
@st.cache_data(show_spinner=False)
def extract_text(_image):

    try:
        img = np.array(_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        )

        text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")

        return text

    except:
        return ""


# ---------------- VALUE EXTRACTION ----------------
def extract_values(text):

    data = {}

    if not text:
        return data

    normalized = re.sub(r"[\n\r\t]+", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized)

    patterns = {
        "Glucose": [
            r"(?:glucose|blood glucose|plasma glucose|glu|fpg|fbs|fasting blood sugar|fasting plasma glucose|blood sugar|sugar|rbs|random blood sugar|ppbs|post prandial blood sugar)\D{0,35}(\d{2,3}(?:\.\d+)?)",
        ],
        "BloodPressure": [
            r"(?:blood pressure|bp|b\.p\.|b p)\D{0,25}(\d{2,3})\s*(?:/|over|-)\s*(\d{2,3})",
            r"(?:systolic|sys)\D{0,25}(\d{2,3})",
        ],
        "BMI": [
            r"(?:bmi|body mass index)\D{0,25}(\d{1,2}(?:\.\d+)?)",
        ],
        "Age": [
            r"(?:age|age/sex|years old|yrs old|y/o)\D{0,20}(\d{1,3})",
            r"(\d{1,3})\s*(?:years|yrs)\b",
            r"\b(\d{1,3})\s*/\s*(?:m|f|male|female)\b",
        ],
    }

    for key, key_patterns in patterns.items():
        for pattern in key_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue

            try:
                data[key] = float(match.group(1))
                break
            except (TypeError, ValueError):
                continue

    return data


def extract_report_items(text):
    items = []

    if not text:
        return items

    known_tests = [
        "glucose",
        "blood sugar",
        "fasting blood sugar",
        "random blood sugar",
        "fbs",
        "rbs",
        "ppbs",
        "hba1c",
        "hemoglobin",
        "cholesterol",
        "triglycerides",
        "hdl",
        "ldl",
        "vldl",
        "creatinine",
        "urea",
        "uric acid",
        "bilirubin",
        "sgpt",
        "sgot",
        "tsh",
        "t3",
        "t4",
        "bmi",
        "blood pressure",
        "bp",
        "age",
    ]

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if len(line) < 3:
            continue

        lower_line = line.lower()
        if not any(test in lower_line for test in known_tests):
            continue

        value_match = re.search(
            r"(?<![A-Za-z])(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?)",
            line,
        )
        if not value_match:
            continue

        value = value_match.group(1).replace(" ", "")
        name = line[: value_match.start()].strip(" :-|")
        trailing = line[value_match.end() :].strip()

        unit_match = re.search(r"([a-zA-Z/%]+(?:/[a-zA-Z]+)?)", trailing)
        unit = unit_match.group(1) if unit_match else ""

        status = "Review"
        if re.search(r"\b(high|positive|raised|elevated|abnormal)\b", lower_line):
            status = "High"
        elif re.search(r"\b(low|reduced|below)\b", lower_line):
            status = "Low"
        elif re.search(r"\b(normal|negative|within)\b", lower_line):
            status = "Normal"

        items.append(
            {
                "Test": name.title() if name else "Detected value",
                "Value": value,
                "Unit": unit,
                "Status": status,
                "Source line": line,
            }
        )

    return items[:20]
