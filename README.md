# AI Medical Report Analyzer

AI Medical Report Analyzer is a Streamlit-based application that reads medical reports using OCR and predicts diabetes risk using machine learning.

## Features

- Upload PDF or image reports
- Extract text using OCR
- Detect:
  - Glucose
  - Blood Pressure
  - BMI
  - Age
- Predict diabetes risk
- Generate doctor-style summary
- AI medical chatbot support

---

## Technologies Used

- Python
- Streamlit
- OpenCV
- Tesseract OCR
- Scikit-learn
- Pandas
- Matplotlib
- OpenAI API

---

## Project Files

```bash
app.py          # Main application
ui.py           # User interface
utils.py        # OCR and value extraction
model.pkl       # Trained ML model
train_model.ipynb