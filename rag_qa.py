import os
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_documents(doc_folder='docs'):
    if not os.path.exists(doc_folder):
        os.makedirs(doc_folder)
        print(f"✅ Created '{doc_folder}' folder. Please add .txt or .pdf files to it.")
        return []

    docs = []
    for filename in os.listdir(doc_folder):
        path = os.path.join(doc_folder, filename)
        if filename.endswith('.txt'):
            loader = TextLoader(path)
            docs.extend(loader.load())
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(path)
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
