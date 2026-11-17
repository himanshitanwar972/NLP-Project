
import streamlit as st
import pandas as pd
import joblib
import nltk
import re
import html

from pypdf import PdfReader
from io import BytesIO

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Keyword Extraction | NLP Studio",
    page_icon="🔑",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS - UI DESIGN ONLY
# --------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(135deg, #0b1120 0%, #111827 55%, #17152e 100%);
    color: #f8fafc;
}

/* Main content width and spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1e1b3a);
    border-right: 1px solid #303653;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

/* Headings */
h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

p, label {
    color: #cbd5e1;
}

/* Main title */
.hero-title {
    font-size: 38px;
    font-weight: 800;
    line-height: 1.2;
    background: linear-gradient(90deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}

.hero-subtitle {
    color: #a5b4fc;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Section headings */
.section-heading {
    font-size: 21px;
    font-weight: 700;
    color: #e9d5ff;
    margin-bottom: 12px;
}

/* Information cards */
.info-card {
    background: linear-gradient(135deg, #1e293b, #252044);
    border: 1px solid #42416b;
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
    margin-bottom: 15px;
}

.info-card h3 {
    font-size: 17px;
    margin-bottom: 8px;
    color: #c4b5fd !important;
}

.info-card p {
    font-size: 13px;
    color: #cbd5e1;
    margin-bottom: 0;
}

/* Input area */
.stTextArea textarea {
    background-color: #172033 !important;
    color: #f8fafc !important;
    border: 1px solid #55517e !important;
    border-radius: 12px !important;
    font-size: 14px !important;
}

/* Select boxes and other controls */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
    background-color: #1e293b !important;
    color: white !important;
    border-color: #55517e !important;
    border-radius: 10px !important;
}

/* Radio buttons */
.stRadio label {
    color: #e2e8f0 !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    color: #a78bfa;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white !important;
    border: none;
    border-radius: 11px;
    padding: 12px 25px;
    font-weight: 700;
    min-height: 48px;
    transition: 0.25s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6d28d9, #1d4ed8);
    box-shadow: 0 5px 20px rgba(124, 58, 237, 0.35);
    transform: translateY(-2px);
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(90deg, #059669, #0d9488);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 20px;
}

.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #047857, #0f766e);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #172033;
    border: 1px dashed #8b7bd8;
    border-radius: 14px;
    padding: 15px;
}

/* Result container */
.result-panel {
    background: linear-gradient(135deg, #1e293b, #242044);
    border: 1px solid #4c477e;
    border-radius: 16px;
    padding: 22px;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Keyword tags */
.keyword-tag {
    display: inline-block;
    background: linear-gradient(135deg, #4338ca, #6d28d9);
    color: white;
    border: 1px solid #8b5cf6;
    padding: 9px 15px;
    margin: 5px;
    border-radius: 25px;
    font-size: 13px;
    font-weight: 600;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #45416e;
    border-radius: 13px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #c4b5fd !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #45416e;
    border-radius: 12px;
    overflow: hidden;
}

/* Divider */
hr {
    border-color: #343653;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding-top: 30px;
    padding-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# NLTK RESOURCES
# --------------------------------------------------

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)


# --------------------------------------------------
# LOAD SAVED TF-IDF VECTORIZER
# --------------------------------------------------

tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_features = tfidf_vectorizer.get_feature_names_out()


# --------------------------------------------------
# INITIALIZE PREPROCESSING TOOLS
# --------------------------------------------------

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# PREPROCESSING FUNCTION
# --------------------------------------------------

def preprocess_text(text):
    text = str(text).lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


# --------------------------------------------------
# TF-IDF EXTRACTION
# --------------------------------------------------

def extract_tfidf(text, top_n):
    processed_text = preprocess_text(text)
    vector = tfidf_vectorizer.transform([processed_text])
    scores = vector.toarray()[0]

    indices = scores.argsort()[::-1]

    results = []

    for index in indices:
        if scores[index] > 0:
            results.append({
                "Keyword": tfidf_features[index],
                "Score": round(float(scores[index]), 4)
            })

        if len(results) == top_n:
            break

    return results


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(BytesIO(uploaded_file.getvalue()))

    extracted_text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text += page_text + "\n"

    return extracted_text.strip()


# --------------------------------------------------
# RAKE EXTRACTION
# --------------------------------------------------

def extract_rake(text, top_n):
    rake = Rake()
    rake.extract_keywords_from_text(text)

    phrases = rake.get_ranked_phrases_with_scores()

    return [
        {
            "Keyword / Keyphrase": phrase,
            "Score": round(float(score), 2)
        }
        for score, phrase in phrases[:top_n]
    ]


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("# 🔑 NLP Studio")
    st.markdown("### Keyword Extraction System")

    st.markdown("---")

    st.markdown("### 📌 About the Project")

    st.write(
        "An NLP-based application that identifies "
        "important keywords and keyphrases from "
        "research documents."
    )

    st.markdown("---")

    st.markdown("### ⚙️ Available Techniques")

    st.markdown("""
    - **TF-IDF:** Statistical keyword extraction
    - **RAKE:** Rapid Automatic Keyword Extraction
    """)

    st.markdown("---")

    st.markdown("### 📄 Supported Input")

    st.markdown("""
    - Research abstract text
    - PDF research papers
    """)

    st.markdown("---")

    st.caption("Developed as an NLP project")



# ==================================================
# MAIN HEADER
# ==================================================

st.html("""
<div style="
    text-align: center;
    padding: 20px 10px 35px 10px;
">
    <div style="font-size: 55px; margin-bottom: 10px;">
        🔑
    </div>

    <h1 style="
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    ">
        Keyword & Keyphrase Extraction
    </h1>

    <p style="
        color: #a5b4fc;
        font-size: 16px;
        margin-top: 0;
    ">
        Intelligent NLP-powered research document analysis
    </p>
</div>
""")


# ==================================================
# TOP INFORMATION CARDS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>📝 Text Analysis</h3>
        <p>Enter research abstracts and identify their most important terms.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>📄 PDF Support</h3>
        <p>Upload research papers and automatically extract their text.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 NLP Techniques</h3>
        <p>Use TF-IDF and RAKE to discover relevant keywords and phrases.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# ==================================================
# INPUT SECTION
# ==================================================

st.markdown(
    '<div class="section-heading">📥 Input Research Content</div>',
    unsafe_allow_html=True
)

st.write("Choose how you want to provide your research content.")

input_method = st.radio(
    "Choose input method:",
    ["Enter Text", "Upload PDF"],
    horizontal=True
)

user_text = ""

if input_method == "Enter Text":

    with st.container(border=True):

        st.markdown("### 📝 Enter Your Research Text")

        user_text = st.text_area(
            "Paste your research abstract here:",
            height=250,
            placeholder=(
                "Paste your research abstract or document content here..."
            ),
            label_visibility="collapsed"
        )

        if user_text.strip():
            st.caption(
                f"Characters entered: {len(user_text):,}"
            )

else:

    with st.container(border=True):

        st.markdown("### 📄 Upload Your Research Paper")

        uploaded_file = st.file_uploader(
            "Upload your research paper (PDF)",
            type=["pdf"],
            help="Select a PDF research paper from your computer."
        )

        if uploaded_file is not None:

            try:
                user_text = extract_text_from_pdf(uploaded_file)

                if user_text:

                    st.success("✅ PDF text extracted successfully!")

                    st.caption(
                        f"Extracted characters: {len(user_text):,}"
                    )

                    with st.expander("👁️ Preview Extracted PDF Text"):
                        st.text_area(
                            "Extracted PDF Text:",
                            value=user_text,
                            height=200,
                            disabled=True
                        )

                else:
                    st.warning(
                        "No extractable text found in this PDF."
                    )

            except Exception as e:
                st.error(f"Error reading PDF: {e}")


st.markdown("---")


# ==================================================
# EXTRACTION SETTINGS
# ==================================================

st.markdown(
    '<div class="section-heading">⚙️ Extraction Settings</div>',
    unsafe_allow_html=True
)

setting_col1, setting_col2 = st.columns(2)

with setting_col1:

    method = st.selectbox(
        "Select Extraction Method:",
        ["TF-IDF", "RAKE"],
        help="Choose the NLP technique for extracting keywords."
    )

with setting_col2:

    top_n = st.slider(
        "Number of keywords to extract:",
        min_value=5,
        max_value=20,
        value=10
    )


if method == "TF-IDF":
    st.info(
        "📊 TF-IDF identifies important individual terms "
        "based on their statistical importance."
    )
else:
    st.info(
        "🧠 RAKE identifies meaningful multi-word "
        "keyphrases from the research text."
    )


st.markdown("---")


# ==================================================
# KEYWORD EXTRACTION
# ==================================================

st.markdown(
    '<div class="section-heading">🚀 Generate Results</div>',
    unsafe_allow_html=True
)

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:

    extract_button = st.button(
        "✨ Extract Keywords",
        use_container_width=True
    )


if extract_button:

    if not user_text.strip():

        st.warning(
            "⚠️ Please enter some research text or upload a PDF first."
        )

    else:

        try:

            with st.spinner("Analyzing your research content..."):

                if method == "TF-IDF":
                    results = extract_tfidf(user_text, top_n)

                else:
                    results = extract_rake(user_text, top_n)

            if results:

                st.success(
                    "✅ Keywords extracted successfully!"
                )

                st.markdown("---")

                st.markdown(
                    '<div class="section-heading">📊 Extraction Results</div>',
                    unsafe_allow_html=True
                )

                result_df = pd.DataFrame(results)

                # Result summary metrics
                metric1, metric2, metric3 = st.columns(3)

                with metric1:
                    st.metric(
                        "Keywords Found",
                        len(results)
                    )

                with metric2:
                    st.metric(
                        "Extraction Method",
                        method
                    )

                with metric3:
                    st.metric(
                        "Input Characters",
                        f"{len(user_text):,}"
                    )

                # Keyword visual display
                st.markdown("### 🏷️ Important Keywords")

                keyword_column = (
                    "Keyword" if method == "TF-IDF"
                    else "Keyword / Keyphrase"
                )

                keyword_tags = ""

                for keyword in result_df[keyword_column]:

                    safe_keyword = html.escape(str(keyword))

                    keyword_tags += (
                        f'<span class="keyword-tag">'
                        f'{safe_keyword}</span>'
                    )

                st.markdown(
                    f'<div class="result-panel">'
                    f'{keyword_tags}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Detailed results table
                st.markdown("### 📋 Detailed Keyword Scores")

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                # CSV download
                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.markdown("### 📥 Export Your Results")

                st.download_button(
                    label="⬇️ Download Keywords as CSV",
                    data=csv_data,
                    file_name="extracted_keywords.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            else:

                st.info(
                    "No keywords found. Try entering a longer "
                    "research abstract."
                )

        except Exception as e:

            st.error(f"An error occurred: {e}")


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <p>🔑 Keyword & Keyphrase Extraction System</p>
    <p>Built using Python, Streamlit and Natural Language Processing</p>
</div>
""", unsafe_allow_html=True)

