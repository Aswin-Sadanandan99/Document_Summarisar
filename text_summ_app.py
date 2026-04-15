import streamlit as st
import requests
import time
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize

# Download once (safe if already present)
nltk.download('punkt_tab')

# ------------------- CONFIG -------------------
API_URL = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"
HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ------------------- FUNCTIONS -------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text


def chunk_text(text, max_chunk_chars=1200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def clean_summary(summary):
    sentences = sent_tokenize(summary)
    if not summary.strip().endswith((".", "!", "?")) and len(sentences) > 1:
        return " ".join(sentences[:-1])
    return summary


def summarize(text):
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 120,
            "min_length": 40,
            "length_penalty": 1.5,
            "num_beams": 6,
            "early_stopping": True
        }
    }

    r = requests.post(API_URL, headers=headers, json=payload)

    if r.status_code != 200:
        st.error(f"Error: {r.status_code}")
        st.text(r.text)
        return None

    data = r.json()

    # Cold start handling
    if isinstance(data, dict) and "estimated_time" in data:
        time.sleep(data["estimated_time"])
        return summarize(text)

    raw_summary = data[0]["summary_text"]
    return clean_summary(raw_summary)


# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="PDF/Text Summarizer", layout="wide")
st.title("📄 PDF & Text Summarizer using Hugging Face")

option = st.radio("Choose Input Type:", ["Upload PDF", "Paste Text"])

text_input = ""

if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text_input = extract_text_from_pdf(uploaded_file)

elif option == "Paste Text":
    text_input = st.text_area("Paste your text here", height=300)


# ------------------- SUMMARIZATION -------------------
if st.button("Summarize"):
    if not text_input.strip():
        st.warning("Please provide text or upload a PDF.")
    else:
        with st.spinner("Summarizing... Please wait"):
            chunks = chunk_text(text_input)

            final_summary = ""
            for chunk in chunks:
                summary = summarize(chunk)
                if summary:
                    final_summary += summary + "\n"

        st.subheader("✅ Summary")
        st.write(final_summary)
