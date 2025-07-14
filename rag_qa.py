import os
import tempfile
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders.base import BaseLoader

def load_documents(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        suffix = uploaded_file.name.split('.')[-1]

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load depending on type
        if suffix == "txt":
            loader = TextLoader(tmp_path)
        elif suffix == "pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        docs.extend(loader.load())
    return docs


def build_vector_store(docs):
    if not docs:
        raise ValueError("❌ No documents found to build vector store.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Free embedding model
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    # Load free LLM (e.g., Flan-T5)
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain
