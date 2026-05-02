import streamlit as st


def render_ui():
    st.markdown(
        """
        <section class="hero">
            <div>
                <span class="eyebrow">Clinical document intelligence</span>
                <h1>AI Medical Report Analyzer</h1>
                <p>
                    Upload a lab report or prescription image to extract report text,
                    estimate diabetes risk, and generate a short clinical summary.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Upload report")
        st.caption("Accepted formats: PDF reports, PNG, JPG, and JPEG images.")
        uploaded_file = st.file_uploader(
            "Choose a PDF or image report",
            type=["png", "jpg", "jpeg", "pdf"],
        )

        analyze = st.button("Analyze report", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="panel side-panel">
                <h3>What this tool checks</h3>
                <div class="check-row"><span>01</span><p>Reads PDF reports and image reports with OCR.</p></div>
                <div class="check-row"><span>02</span><p>Finds glucose, blood pressure, BMI, and age when present.</p></div>
                <div class="check-row"><span>03</span><p>Predicts diabetes risk with mean values for missing fields.</p></div>
                <div class="check-row"><span>04</span><p>Creates a concise doctor-style explanation.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="notice">
            This app supports decision-making and education only. Always consult a
            qualified medical professional for diagnosis or treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return analyze, uploaded_file
