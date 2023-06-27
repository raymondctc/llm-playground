from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings.fake import FakeEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import UnstructuredFileLoader

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

        # split into chunks
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=100,
            chunk_overlap=50,
            length_function=len
        )
        chunks = splitter.split_text(text)
        st.write(f"Number of chunks: {len(chunks)}")
        st.write(chunks)

        # create embeddings
        embeddings = FakeEmbeddings(size=1352)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about your document:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            st.write(docs)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback as cb:
                response = chain.run(input_documents=docs, question=user_question)
            st.write(response)

if __name__ == "__main__":
    main()