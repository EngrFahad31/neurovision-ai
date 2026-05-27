# =========================================================
# 🧠 NEUROVISION AI
# FINAL PREMIUM EEG SEIZURE DETECTION SYSTEM
# Powered By Engr Fahad Naseer
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import pywt
import joblib
import tempfile
import time

from scipy.signal import stft

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="NeuroVision AI",

    page_icon="🧠",

    layout="wide"
)

# =========================================================
# PROFESSIONAL CSS
# =========================================================
st.markdown("""
<style>

/* =========================================================
🔥 ULTRA FUTURISTIC NEURO UI
========================================================= */

/* MAIN BACKGROUND */

.stApp{

    background:
    radial-gradient(circle at top,
    #111827 0%,
    #020617 40%,
    #000000 100%);

    color:white;
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"]{

    background:
    linear-gradient(
    180deg,
    #000000,
    #111827
    );

    border-right:2px solid #38BDF8;

    box-shadow:
    0px 0px 20px rgba(56,189,248,0.3);
}

/* =========================================================
HEADINGS
========================================================= */

h1,h2,h3,h4{

    color:#38BDF8;

    font-weight:bold;

    letter-spacing:1px;
}

/* =========================================================
METRIC CARDS
========================================================= */

[data-testid="metric-container"]{

    background:
    linear-gradient(
    180deg,
    rgba(17,24,39,0.95),
    rgba(30,41,59,0.95)
    );

    border:1px solid rgba(56,189,248,0.5);

    border-radius:18px;

    padding:18px;

    box-shadow:
    0px 0px 20px rgba(56,189,248,0.15);

    transition:0.3s;
}

/* HOVER EFFECT */

[data-testid="metric-container"]:hover{

    transform:translateY(-4px);

    border:1px solid #38BDF8;

    box-shadow:
    0px 0px 30px rgba(56,189,248,0.4);
}

/* =========================================================
BUTTONS
========================================================= */

div.stButton > button{

    background:
    linear-gradient(
    90deg,
    #0EA5E9,
    #2563EB
    );

    color:white;

    border:none;

    border-radius:12px;

    padding:10px 22px;

    font-size:16px;

    font-weight:bold;

    transition:0.3s;

    box-shadow:
    0px 0px 15px rgba(37,99,235,0.4);
}

/* BUTTON HOVER */

div.stButton > button:hover{

    transform:scale(1.02);

    box-shadow:
    0px 0px 25px rgba(56,189,248,0.6);
}

/* =========================================================
INPUT BOXES
========================================================= */

.stTextInput input,
.stTextArea textarea{

    background-color:#111827;

    color:white;

    border-radius:10px;

    border:1px solid rgba(56,189,248,0.4);
}

/* =========================================================
SELECT BOX
========================================================= */

.stSelectbox div{

    background-color:#111827;

    color:white;

    border-radius:10px;
}

/* =========================================================
FILE UPLOADER
========================================================= */

section[data-testid="stFileUploader"]{

    background:
    rgba(17,24,39,0.9);

    border:2px dashed rgba(56,189,248,0.5);

    border-radius:16px;

    padding:20px;

    box-shadow:
    0px 0px 20px rgba(56,189,248,0.15);
}

/* =========================================================
TABS
========================================================= */

.stTabs [data-baseweb="tab"]{

    background:#111827;

    color:white;

    border-radius:10px;

    padding:10px;

    margin-right:8px;

    border:1px solid rgba(56,189,248,0.2);
}

/* ACTIVE TAB */

.stTabs [aria-selected="true"]{

    background:
    linear-gradient(
    90deg,
    #0EA5E9,
    #2563EB
    ) !important;

    color:white !important;

    font-weight:bold;

    box-shadow:
    0px 0px 15px rgba(56,189,248,0.4);
}

/* =========================================================
PROGRESS BAR
========================================================= */

.stProgress > div > div > div > div{

    background:
    linear-gradient(
    90deg,
    #0EA5E9,
    #2563EB
    );
}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"]{

    border-radius:12px;

    overflow:hidden;
}

/* =========================================================
FOOTER CARD STYLE
========================================================= */

.footer-card{

    background:#111827;

    border:1px solid rgba(56,189,248,0.4);

    border-radius:18px;

    padding:20px;

    text-align:center;

    box-shadow:
    0px 0px 20px rgba(56,189,248,0.15);
}

</style>
""", unsafe_allow_html=True)
# =========================================================
# HERO SECTION
# =========================================================

# =========================================================
# CLEAN HERO SECTION (NO HTML BUG)
# =========================================================

hero_col1, hero_col2, hero_col3 = st.columns([1,6,1])

with hero_col2:

    st.markdown("")

    st.title("🧠 NeuroVision AI")

    st.subheader(
        "Advanced EEG Seizure Detection & Brainwave Intelligence"
    )

    st.write(
        "AI Powered Neurological Diagnostic Platform"
    )

    st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🧠 NeuroVision AI"
)

st.sidebar.success(
    "System Online"
)

st.sidebar.info(
    "AI Diagnostic Active"
)

st.sidebar.warning(
    "EEG Monitoring Enabled"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "AI Modules"
)

st.sidebar.write("• FFT Analysis")
st.sidebar.write("• STFT Analysis")
st.sidebar.write("• Wavelet Transform")
st.sidebar.write("• Entropy Analysis")
st.sidebar.write("• Random Forest AI")

st.sidebar.markdown("---")

st.sidebar.write(
    "Powered By Engr Fahad Naseer"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "final_eeg_seizure_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

fs = 256

# =========================================================
# PATIENT INFO
# =========================================================

st.subheader(
    "👤 Patient Information"
)

c1,c2 = st.columns(2)

with c1:

    patient_name = st.text_input(
        "Patient Name"
    )

    patient_age = st.text_input(
        "Age"
    )

    patient_gender = st.selectbox(
        "Gender",
        ["Male","Female","Other"]
    )

    patient_id = st.text_input(
        "Patient ID"
    )

with c2:

    doctor_name = st.text_input(
        "Doctor Name"
    )

    hospital_name = st.text_input(
        "Hospital / Clinic"
    )

    doctor_notes = st.text_area(
        "Doctor Notes"
    )

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def spectral_entropy(signal):

    fft_vals = np.abs(
        np.fft.rfft(signal)
    ) ** 2

    psd = fft_vals / np.sum(fft_vals)

    entropy = -np.sum(
        psd * np.log2(psd + 1e-12)
    )

    return entropy

def extract_features(segment):

    features = []

    for ch in segment:

        mean = np.mean(ch)

        std = np.std(ch)

        energy = np.sum(ch ** 2)

        fft_vals = np.abs(
            np.fft.rfft(ch)
        )

        fft_mean = np.mean(fft_vals)

        fft_std = np.std(fft_vals)

        entropy = spectral_entropy(ch)

        f,t,Zxx = stft(
            ch,
            fs=fs
        )

        stft_energy = np.sum(
            np.abs(Zxx) ** 2
        )

        coeffs = pywt.wavedec(
            ch,
            'db4',
            level=4
        )

        wave_features = []

        for c in coeffs:

            wave_features.extend([

                np.mean(c),

                np.std(c),

                np.sum(c ** 2)
            ])

        features.extend([

            mean,
            std,
            energy,

            fft_mean,
            fft_std,

            entropy,

            stft_energy
        ])

        features.extend(
            wave_features
        )

    return features

# =========================================================
# PDF STYLE
# =========================================================

def add_pdf_style(canvas, doc):

    canvas.saveState()

    width, height = A4

    # BORDER

    canvas.setStrokeColor(colors.black)

    canvas.setLineWidth(2)

    canvas.rect(
        20,
        20,
        width - 40,
        height - 40
    )

    # HEADER

    canvas.setFont(
        "Helvetica-Bold",
        16
    )

    canvas.drawString(
        40,
        height - 40,
        "NeuroVision AI Clinical EEG Report"
    )

    # FOOTER

    canvas.setFont(
        "Helvetica",
        10
    )

    canvas.drawString(
        40,
        30,
        "Powered By Engr Fahad Naseer"
    )

    # WATERMARK

    canvas.setFont(
        "Helvetica",
        35
    )

    canvas.setFillGray(
        0.92,
        0.2
    )

    canvas.drawCentredString(

        300,

        420,

        "NeuroVision AI"
    )

    canvas.restoreState()

# =========================================================
# PDF REPORT
# =========================================================

def create_pdf(

    seizure_found,

    seizure_time,

    confidence,

    brain_score,

    total_segments,

    channels
):

    doc = SimpleDocTemplate(

        "Clinical_EEG_Report.pdf",

        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = []

    # =====================================================
    # TITLE
    # =====================================================

    title = Paragraph(

        "<b>Clinical EEG Analysis Report</b>",

        styles['Title']
    )

    elements.append(title)

    elements.append(
        Spacer(1,20)
    )

    # =====================================================
    # PATIENT DETAILS
    # =====================================================

    patient_table_data = [

        ["Patient Name", patient_name],

        ["Patient ID", patient_id],

        ["Age", patient_age],

        ["Gender", patient_gender],

        ["Doctor", doctor_name],

        ["Hospital", hospital_name]
    ]

    patient_table = Table(

        patient_table_data,

        colWidths=[200,250]
    )

    patient_table.setStyle(

        TableStyle([

            ('GRID',(0,0),(-1,-1),1,colors.black),

            ('BACKGROUND',(0,0),(0,-1),colors.lightgrey),

            ('BOTTOMPADDING',(0,0),(-1,-1),10),

            ('TOPPADDING',(0,0),(-1,-1),10),
        ])
    )

    elements.append(patient_table)

    elements.append(
        Spacer(1,25)
    )

    # =====================================================
    # RESULTS
    # =====================================================

    if seizure_found:

        result_text = f"""

        <b>EEG Result:</b><br/><br/>

        Abnormal seizure-related neurological activity detected.<br/><br/>

        <b>Detection Time:</b>
        {seizure_time:.2f} seconds<br/><br/>

        <b>AI Confidence:</b>
        {confidence:.2f}%<br/><br/>

        <b>Brain Stability Score:</b>
        {brain_score}/100
        """

    else:

        result_text = f"""

        <b>EEG Result:</b><br/><br/>

        EEG activity appears neurologically stable.<br/><br/>

        <b>Brain Stability Score:</b>
        {brain_score}/100
        """

    result_para = Paragraph(

        result_text,

        styles['BodyText']
    )

    elements.append(result_para)

    elements.append(
        Spacer(1,25)
    )

    # =====================================================
    # TECHNICAL SUMMARY
    # =====================================================

    tech_title = Paragraph(

        "<b>Technical Summary</b>",

        styles['Heading2']
    )

    elements.append(tech_title)

    elements.append(
        Spacer(1,10)
    )

    tech_data = [

        ["Sampling Frequency", "256 Hz"],

        ["EEG Channels", str(channels)],

        ["Analyzed Segments", str(total_segments)],

        ["AI Model", "Random Forest"],

        ["Features", "FFT + STFT + Wavelet"],

        ["Filtering", "0.5 - 40 Hz Bandpass"]
    ]

    tech_table = Table(

        tech_data,

        colWidths=[220,220]
    )

    tech_table.setStyle(

        TableStyle([

            ('GRID',(0,0),(-1,-1),1,colors.black),

            ('BACKGROUND',(0,0),(0,-1),colors.lightgrey),

            ('BOTTOMPADDING',(0,0),(-1,-1),10),

            ('TOPPADDING',(0,0),(-1,-1),10),
        ])
    )

    elements.append(tech_table)

    elements.append(
        Spacer(1,25)
    )

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    recommendation = Paragraph(

        """
        <b>Recommendations</b><br/><br/>

        • Neurological consultation recommended.<br/>
        • Follow-up EEG monitoring advised.<br/>
        • Clinical correlation suggested.<br/>
        • Results should be reviewed by a neurologist.
        """,

        styles['BodyText']
    )

    elements.append(recommendation)

    elements.append(
        Spacer(1,25)
    )

    # =====================================================
    # NOTES
    # =====================================================

    notes = Paragraph(

        f"""
        <b>Doctor Notes</b><br/><br/>
        {doctor_notes}
        """,

        styles['BodyText']
    )

    elements.append(notes)

    elements.append(
        Spacer(1,40)
    )

    # =====================================================
    # SIGNATURE
    # =====================================================

    signature = Paragraph(

        """
        ___________________________<br/>
        Doctor Signature
        """,

        styles['BodyText']
    )

    elements.append(signature)

    doc.build(

        elements,

        onFirstPage=add_pdf_style,

        onLaterPages=add_pdf_style
    )

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(

    "📂 Upload EDF EEG File",

    type=["edf"]
)

# =========================================================
# MAIN PROCESSING
# =========================================================

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".edf"

    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        temp_path = tmp.name

    progress = st.progress(0)

    messages = [

        "Initializing EEG Scanner...",

        "Filtering EEG Signals...",

        "Extracting Biomedical Features...",

        "Running AI Detection...",

        "Generating Clinical Report..."
    ]

    for i,msg in enumerate(messages):

        progress.progress((i+1)*20)

        st.info(msg)

        time.sleep(0.7)

    # =====================================================
    # LOAD EEG
    # =====================================================

    raw = mne.io.read_raw_edf(

        temp_path,

        preload=True,

        verbose=False
    )

    raw.filter(
        0.5,
        40
    )

    data = raw.get_data()

    # =====================================================
    # SEGMENTATION
    # =====================================================

    window_size = 1280

    seizure_found = False

    seizure_time = 0

    confidence = 0

    seizure_zone = None

    preds = []

    total_segments = 0

    for start in range(

        0,

        data.shape[1] - window_size,

        window_size
    ):

        end = start + window_size

        segment = data[:,start:end]

        total_segments += 1

        features = extract_features(
            segment
        )

        features = np.array(
            features
        ).reshape(1,-1)

        scaled = scaler.transform(
            features
        )

        pred = model.predict(
            scaled
        )

        prob = model.predict_proba(
            scaled
        )[0][1]

        preds.append(prob * 100)

        if pred[0] == 1:

            seizure_found = True

            seizure_time = start / fs

            confidence = prob * 100

            seizure_zone = (start,end)

            break

    # =====================================================
    # SCORE
    # =====================================================

    if seizure_found:

        brain_score = int(
            100 - confidence
        )

    else:

        brain_score = 95

    # =====================================================
    # CREATE PDF
    # =====================================================

    create_pdf(

        seizure_found,

        seizure_time,

        confidence,

        brain_score,

        total_segments,

        data.shape[0]
    )

    # =====================================================
    # DASHBOARD
    # =====================================================

    st.markdown("---")

    m1,m2,m3,m4 = st.columns(4)

    m1.metric(
        "Brain Score",
        f"{brain_score}/100"
    )

    m2.metric(
        "Channels",
        data.shape[0]
    )

    m3.metric(
        "Sampling",
        "256 Hz"
    )

    m4.metric(
        "Risk",
        "HIGH" if seizure_found else "LOW"
    )

    st.markdown("---")

    # =====================================================
    # TABS
    # =====================================================

    tab1,tab2,tab3 = st.tabs([

        "📊 Dashboard",

        "📈 EEG Graphs",

        "🩺 Clinical Report"
    ])

    # =====================================================
    # DASHBOARD
    # =====================================================

    with tab1:

        if seizure_found:

            st.error(
                "⚠ Seizure Activity Detected"
            )

            st.write(
                f"Detection Time: {seizure_time:.2f} sec"
            )

            st.write(
                f"AI Confidence: {confidence:.2f}%"
            )

        else:

            st.success(
                "✅ EEG Appears Neurologically Stable"
            )

        st.subheader(
            "Prediction Timeline"
        )

        df = pd.DataFrame({

            "Segment": range(len(preds)),

            "Probability": preds
        })

        st.line_chart(
            df.set_index("Segment")
        )

    # =====================================================
    # EEG GRAPHS
    # =====================================================

    with tab2:

        st.subheader(
            "Multi-Channel EEG"
        )

        fig,axs = plt.subplots(

            4,

            1,

            figsize=(14,8)
        )

        for i in range(4):

            axs[i].plot(

                data[i][:3000],

                color='cyan'
            )

            if seizure_zone is not None:

                axs[i].axvspan(

                    seizure_zone[0],

                    seizure_zone[1],

                    color='red',

                    alpha=0.3
                )

            axs[i].set_title(
                f"Channel {i+1}"
            )

        plt.tight_layout()

        st.pyplot(fig)

        st.subheader(
            "EEG Heatmap"
        )

        fig2,ax2 = plt.subplots(
            figsize=(14,5)
        )

        sns.heatmap(

            data[:10,:1000],

            cmap="magma",

            ax=ax2
        )

        st.pyplot(fig2)

    # =====================================================
    # REPORT
    # =====================================================

    with tab3:

        if seizure_found:

            st.error(
                "Abnormal neurological activity detected."
            )

        else:

            st.success(
                "EEG Appears Stable"
            )

        with open(

            "Clinical_EEG_Report.pdf",

            "rb"
        ) as pdf_file:

            st.download_button(

                "📄 Download Clinical EEG Report",

                pdf_file,

                file_name="Clinical_EEG_Report.pdf",

                mime="application/pdf"
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""

<div style="
background:#111827;
padding:20px;
border-radius:15px;
text-align:center;
">

<h3 style="
color:#38BDF8;
">
🧠 NeuroVision AI
</h3>

<p style="
color:white;
">
Powered By Engr Fahad Naseer • AI & Biomedical Engineer
</p>

</div>

""", unsafe_allow_html=True)