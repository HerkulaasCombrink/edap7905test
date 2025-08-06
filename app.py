import os
import streamlit as st
import openai
import PyPDF2
from openai.embeddings_utils import get_embedding
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
# OpenAI key via env or Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# Session state initialization
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.text_chunks = []

st.set_page_config(page_title="🔍 RAG QA App", layout="wide")
st.title("🔍 Retrieval-Augmented QA (Custom) with Streamlit")

# Sidebar: Index documents
st.sidebar.header("📄 Document Indexing")
uploaded_files = st.sidebar.file_uploader(
    "Upload txt or PDF", type=["txt", "pdf"], accept_multiple_files=True
)
if st.sidebar.button("Index Documents") and uploaded_files:
    raw_texts = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            raw = "".join([page.extract_text() or '' for page in reader.pages])
        else:
            raw = file.read().decode("utf-8")
        raw_texts.append(raw)
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for txt in raw_texts:
        chunks.extend(splitter.split_text(txt))

    # Generate embeddings and build FAISS index
    embeddings = [get_embedding(c, engine="text-embedding-ada-002") for c in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))

    # Store in session
    st.session_state.faiss_index = index
    st.session_state.text_chunks = chunks
    st.sidebar.success(f"Indexed {len(chunks)} chunks!")

# Main QA interface
st.header("💬 Ask a Question")
if st.session_state.faiss_index is None:
    st.info("Upload and 'Index Documents' in the sidebar first.")
else:
    query = st.text_input("Enter your question:")
    if st.button("Answer me") and query:
        with st.spinner("Retrieving and generating answer…"):
            # Embed query and search
            q_emb = get_embedding(query, engine="text-embedding-ada-002")
            D, I = st.session_state.faiss_index.search(
                np.array([q_emb], dtype='float32'), k=4
            )
            retrieved = [st.session_state.text_chunks[i] for i in I[0]]
            # Compose prompt
            context = "\n---\n".join(retrieved)
            prompt = f"Use the context below to answer the question.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"  
            # Call OpenAI Chat
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that uses provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
            )
            answer = resp.choices[0].message.content
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Source Passages:")
        for chunk in retrieved:
            st.markdown(f"> {chunk[:300]}...")

# Footer
st.markdown("---")
st.write("Run with: `streamlit run app.py`. Requires `openai`, `faiss-cpu`, `PyPDF2`, and `langchain`. ")
