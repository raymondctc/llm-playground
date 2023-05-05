from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader


def main():
    load_dotenv()
    st.set_page_config(page_title="Streamlit App")
    st.header("Streamlit App")

    pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(f"Number of pages: {len(pdf_reader.pages)}")
        st.write(f"Page size: {pdf_reader.pages[0].mediabox.right}")
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        st.write(text)

if __name__ == "__main__":
    main()