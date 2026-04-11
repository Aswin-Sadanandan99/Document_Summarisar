import streamlit as st
import requests
import time
from pypdf import PdfReader

# ------------------- CONFIG -------------------
API_URL = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"
HF_TOKEN = "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"  # <-- put your token here

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ------------------- FUNCTIONS -------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def summarize(text):
    payload = {"inputs": text}

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

    return data[0]["summary_text"]


def chunk_text(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


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
