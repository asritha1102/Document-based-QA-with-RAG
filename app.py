import streamlit as st
import tempfile
from rag_qa import load_documents, build_vector_store, create_qa_chain

st.set_page_config(page_title="📄 RAG QA System", layout="wide")
st.title("📄💬 Document-based QA with RAG")

# File uploader
uploaded_files = st.file_uploader(
    "📤 Upload TXT or PDF files", 
    type=["txt", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("📚 Loading uploaded documents..."):
        try:
            docs = load_documents(uploaded_files)
            vectorstore = build_vector_store(docs)
            qa_chain = create_qa_chain(vectorstore)
            st.success("✅ Documents processed and ready!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()
    
    # Ask question
    question = st.text_input("Ask a question based on the uploaded documents:")

    if question:
        with st.spinner("💡 Generating answer..."):
            try:
                answer = qa_chain.run(question)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"❌ Failed to generate answer: {e}")
else:
    st.info("Please upload one or more documents to begin.")
