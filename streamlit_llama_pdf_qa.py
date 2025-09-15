# streamlit_llama_pdf_qa.py

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
import os

# Step 1: Load PDF
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load_and_split()

# Step 2: Create vectorstore
def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="db")
    vectordb.persist()
    return vectordb

# Step 3: Load LLaMA or Mistral Model
def load_llm():
    llm = LlamaCpp(
        model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Replace with your path
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        n_ctx=2048,
        verbose=False,
    )
    return llm

# Step 4: Create QA Chain
def create_qa_chain(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain

# Step 5: Streamlit UI
st.set_page_config(page_title="📄🦙 PDF Q&A with LLaMA", layout="centered")
st.title("📄🦙 Ask Questions from Your PDF (LLaMA Local)")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file is not None and st.button("Process PDF"):
    with open("notes.pdf", "wb") as f:
        f.write(pdf_file.read())
    with st.spinner("Loading and processing PDF..."):
        docs = load_pdf("notes.pdf")
        vectordb = create_vectorstore(docs)
        llm = load_llm()
        st.session_state.qa_chain = create_qa_chain(llm, vectordb)
    st.success("✅ Ready! Ask your questions below.")

if st.session_state.qa_chain:
    query = st.text_input("Ask a question from your PDF:")
    if query:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain({"query": query})
            st.write("**Answer:**", result["result"])
