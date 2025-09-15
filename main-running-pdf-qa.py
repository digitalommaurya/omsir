import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Step 1: Load PDF
def load_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()
    return docs

# Step 2: Create FAISS vectorstore
def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(texts, embedding=embeddings)
    return vectordb

# Step 3: Load Ollama LLM
def load_llm():
    return OllamaLLM(model="llama3")

# Step 4: Build Retrieval QA Chain
def create_qa_chain(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="📄 PDF Q&A with Ollama", layout="centered")
st.title("📄 Ask Questions About Your PDF")
st.markdown("Powered by **Ollama + LLaMA 3 + FAISS**")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("🔍 Loading and processing PDF..."):
        docs = load_pdf(pdf_path)
        vectordb = create_vectorstore(docs)
        llm = load_llm()
        qa_chain = create_qa_chain(llm, vectordb)

    st.success(f"✅ Processed {len(docs)} document chunks. You can now ask questions.")

    query = st.text_input("💬 Ask a question about the PDF:")

    if query:
        with st.spinner("🤖 Thinking..."):
            try:
                result = qa_chain.invoke({"query": query})
                st.markdown("📘 **Answer:**")
                st.success(result)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
