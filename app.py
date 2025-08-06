import os
import streamlit as st

# Embeddings import compatibility
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    raise ImportError("OpenAIEmbeddings not found: please install the 'langchain-openai' package")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import PyPDF2

# --- Configuration ---
# Set your OpenAI API key via env var or Streamlit secrets
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize session state for FAISS index
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

st.set_page_config(page_title="🔍 RAG QA App", layout="wide")
st.title("🔍 Retrieval-Augmented QA with Streamlit")

# Sidebar: upload & index docs
st.sidebar.header("📄 Document Indexing")
uploaded_files = st.sidebar.file_uploader(
    "Upload txt or PDF", type=["txt", "pdf"], accept_multiple_files=True
)
if st.sidebar.button("Index Documents") and uploaded_files:
    docs = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            text = "".join([p.extract_text() or '' for p in reader.pages])
        else:
            text = file.read().decode("utf-8")
        docs.append(text)
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text("\n".join(docs))
    # Build FAISS index
    index = FAISS.from_texts(chunks, embeddings)
    st.session_state.faiss_index = index
    st.sidebar.success(f"Indexed {len(chunks)} chunks!")

# Main: QA interface
st.header("💬 Ask a Question")
if st.session_state.faiss_index is None:
    st.info("Upload and index documents in the sidebar first.")
else:
    query = st.text_input("Enter your question:")
    if st.button("Answer me") and query:
        with st.spinner("Retrieving relevant passages and generating answer…"):
            retriever = st.session_state.faiss_index.as_retriever(search_kwargs={"k": 4})
            llm = OpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.1)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain(query)
            answer = result["result"]
            sources = result["source_documents"]
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Source Passages:")
        for doc in sources:
            st.markdown(f"> {doc.page_content[:500]}...")
