# 🔑 Keyword and Keyphrase Extraction System

An NLP-based application that automatically extracts important keywords and keyphrases from research papers using **TF-IDF** and **RAKE** algorithms. The system provides an interactive Streamlit interface that allows users to enter research text or upload PDF documents and download the extracted results as CSV files.

---

## 📌 1. Project Overview

Research papers contain large amounts of information, making it difficult to identify their most important concepts quickly.

The **Keyword and Keyphrase Extraction System** aims to simplify this process by using Natural Language Processing (NLP) techniques to identify significant words and phrases from research documents.

The project implements two keyword extraction techniques:

- **TF-IDF (Term Frequency–Inverse Document Frequency):** Identifies important individual terms based on their statistical importance.
- **RAKE (Rapid Automatic Keyword Extraction):** Extracts meaningful keywords and multi-word keyphrases from text.

The application also provides an interactive web interface using Streamlit.

---

## 🎯 2. Project Objectives

- Identify important keywords and keyphrases from research papers.
- Collect and prepare a research-paper dataset.
- Perform text cleaning and preprocessing.
- Explore textual data through EDA and visualizations.
- Implement Bag of Words and TF-IDF feature extraction.
- Develop keyword extraction models using TF-IDF and RAKE.
- Evaluate extraction performance using Precision, Recall, and F1-score.
- Build an interactive web application.
- Support both text input and PDF document uploads.

---

## 📂 3. Dataset Description

The project uses the **Inspec dataset**, obtained from Hugging Face.

Dataset: [Inspec Dataset – Hugging Face](https://huggingface.co/datasets/taln-ls2n/inspec)

The dataset contains scientific research abstracts and their associated keyphrases.

The training split containing 1,000 research abstracts was used for this project.

### Dataset attributes

- `id`: Unique identifier of the research paper.
- `title`: Title of the research paper.
- `abstract`: Research paper abstract.
- `keyphrases`: Reference keywords and keyphrases.
- `prmu`: Additional dataset information.

---

## 🧹 4. Data Cleaning

The dataset was cleaned to improve data quality and consistency.

The following operations were performed:

- Handling missing values.
- Removing duplicate research abstracts.
- Removing empty titles and abstracts.
- Removing unnecessary leading and trailing whitespace.
- Saving the cleaned dataset for further processing.

---

## 🔤 5. Text Preprocessing

Text preprocessing was performed using NLTK and regular expressions.

The following steps were implemented:

1. Convert text into lowercase.
2. Tokenize text into individual words.
3. Remove English stopwords.
4. Apply lemmatization to reduce words to their base forms.

The processed text was saved for further analysis and feature extraction.

---

## 📊 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset and identify textual patterns.

The following analyses and visualizations were created:

- Abstract length analysis.
- Distribution of abstract lengths.
- Most frequently occurring words.
- Boxplot analysis of abstract lengths.

These visualizations helped understand the structure and characteristics of the research abstracts.

---

## ⚙️ 7. Feature Engineering

Two feature extraction techniques were implemented.

### A. Bag of Words (BoW)

Bag of Words converts text into numerical feature vectors based on word occurrence frequency.

A `CountVectorizer` was used with a maximum feature limit of 1,000.

### B. TF-IDF

TF-IDF assigns importance scores to words based on their frequency in a document and their distribution across the document collection.

A `TfidfVectorizer` was implemented with a maximum feature limit of 1,000.

These numerical representations were used as part of the keyword extraction system.

---

## 🧠 8. Model Development

Two keyword extraction approaches were implemented.

### A. TF-IDF Keyword Extraction

TF-IDF calculates the importance of individual terms using their TF-IDF scores.

The system:

- Preprocesses the input text.
- Converts the text into a TF-IDF vector.
- Calculates the importance score of each term.
- Sorts terms in descending order of their scores.
- Returns the requested number of top keywords.

### B. RAKE Keyword Extraction

RAKE identifies important keywords and multi-word keyphrases based on word frequency and co-occurrence patterns.

The system:

- Processes the input text.
- Identifies candidate phrases.
- Calculates phrase scores.
- Ranks phrases according to their scores.
- Returns the top-ranked keyphrases.

---

## 📈 9. Model Evaluation

The implemented extraction approaches were evaluated using the following metrics:

- **Precision:** Measures how many extracted keywords are relevant.
- **Recall:** Measures how many relevant reference keywords are successfully identified.
- **F1-score:** Represents the harmonic mean of Precision and Recall.

Evaluation results were calculated using the dataset's reference keyphrases.

Note: TF-IDF and RAKE use different matching approaches in the evaluation. TF-IDF is evaluated using word-level overlap, whereas RAKE is evaluated using normalized phrase-level matching. Therefore, their scores should be interpreted with this difference in mind.

---

## 🖥️ 10. Graphical User Interface (GUI)

The project includes an interactive web application developed using **Streamlit**.

### Main features

- Text input for research abstracts.
- PDF document upload.
- Automatic text extraction from uploaded PDFs.
- Selection between TF-IDF and RAKE.
- Adjustable number of keywords to extract (5–20).
- Display of extracted keywords and keyphrases.
- Display of corresponding extraction scores.
- Download extracted results in CSV format.

---

## 🛠️ 11. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data handling and preprocessing |
| NLTK | Text preprocessing and linguistic processing |
| Scikit-learn | BoW and TF-IDF feature extraction |
| RAKE-NLTK | Keyword and keyphrase extraction |
| Matplotlib | Data visualization |
| Streamlit | Interactive web application |
| PyPDF | PDF text extraction |
| Joblib | Saving and loading the TF-IDF vectorizer |
| Jupyter Notebook | Project implementation and experimentation |

---

## 📁 12. Project Structure

```text
NLP-Project/
│
├── app.py
├── keyword_extraction.ipynb
├── tfidf_vectorizer.pkl
│
├── research_papers.csv
├── cleaned_research_papers.csv
├── preprocessed_research_papers.csv
├── eda_research_papers.csv
│
├── bow_features.csv
├── tfidf_features.csv
├── keyword_extraction_results.csv
├── model_evaluation_results.csv
│
├── .gitignore
└── README.md
```

---

## 🚀 13. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/himanshitanwar972/NLP-Project.git
```

### Step 2: Navigate to the project directory

```bash
cd NLP-Project
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install the required libraries

```bash
pip install streamlit pandas joblib nltk scikit-learn rake-nltk pypdf matplotlib
```

### Step 6: Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

**Note:** The application requires the saved `tfidf_vectorizer.pkl` file in the project directory. It is included in this repository.

---

## 📄 14. Application Workflow

```text
Research Paper / Abstract
          |
          v
     Data Cleaning
          |
          v
   Text Preprocessing
          |
          v
   Feature Engineering
      /          \
     v            v
   BoW          TF-IDF
                  |
                  v
        Keyword Extraction
           /          \
          v            v
       TF-IDF          RAKE
           \          /
            v        v
          Evaluation
              |
              v
       Streamlit GUI
              |
              v
       Extracted Keywords
              |
              v
          CSV Export
```

---

## 🔮 15. Future Enhancements

- Add OCR support for scanned PDF documents.
- Support additional keyword extraction algorithms.
- Improve phrase-level evaluation.
- Add multilingual keyword extraction.
- Explore embedding-based keyword extraction techniques.

---

## 👩‍💻 16. Author

**Project:** Keyword and Keyphrase Extraction System  
**Domain:** Natural Language Processing (NLP)  
**Programming Language:** Python

---

## 🔗 GitHub Repository

[https://github.com/himanshitanwar972/NLP-Project](https://github.com/himanshitanwar972/NLP-Project)

---

