import streamlit as st
from rag_qa import load_documents, build_vector_store, create_qa_chain
from dotenv import load_dotenv
import os

load_dotenv()  # Loads the .env file

st.set_page_config(page_title="📄 RAG QA System", layout="wide")
st.title("📄💬 Document-based QA with RAG")

with st.spinner("📚 Loading documents..."):
    try:
        docs = load_documents()
        if not docs:
            st.warning("No documents found in the `docs/` folder.")
        else:
            vectorstore = build_vector_store(docs)
            qa_chain = create_qa_chain(vectorstore)
            st.success("✅ Ready to answer your questions!")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

question = st.text_input("Ask a question based on your documents:")

if question and 'qa_chain' in locals():
    with st.spinner("💡 Generating answer..."):
        try:
            answer = qa_chain.run(question)
            st.markdown(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"❌ Failed to generate answer: {e}")
else:
    st.info("Enter a question after documents are loaded.")
