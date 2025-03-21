import os
import glob
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import ollama
import streamlit as st

logging.basicConfig(level=logging.INFO)

PDF_DIRECTORY = r"C:/Users/ankam/Documents/pdf_folder"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_DIR = "./chroma_db"

# Load and process PDFs
def ingest_pdfs(pdf_directory):
    all_docs = []
    for pdf_file in glob.glob(os.path.join(pdf_directory, '*.pdf')):
        loader = PyMuPDFLoader(pdf_file)
        all_docs.extend(loader.load())
    logging.info(f"Loaded {len(all_docs)} documents.")
    return all_docs

# Split documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

# Create or load vector database
def get_vector_db(chunks):
    ollama.pull(EMBEDDING_MODEL)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=VECTOR_STORE_DIR
    )
    vector_db.persist()
    logging.info("Vector database ready and persisted locally.")
    return vector_db

# Setup RetrievalQA chain
def setup_qa_chain(vector_db):
    llm = ChatOllama(model=MODEL_NAME)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    prompt_template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    logging.info("QA chain created successfully.")
    return qa_chain

# Streamlit App

def main():
    st.title("📚 UToledo IT Helpdesk Chatbot")

    # Initialize once
    if "qa_chain" not in st.session_state:
        with st.spinner("Loading and preparing documents..."):
            docs = ingest_pdfs(PDF_DIRECTORY)
            if not docs:
                st.error("No PDFs found in the specified directory.")
                return
            chunks = split_documents(docs)
            vector_db = get_vector_db(chunks)
            st.session_state.qa_chain = setup_qa_chain(vector_db)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.write(content)

    user_input = st.chat_input("Ask your IT question here:")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.qa_chain.invoke({"query": user_input})
                    answer = response["result"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Optional: Display source documents
                    #with st.expander("Source Documents"):
                       #for doc in response["source_documents"]:
                            #st.write(doc.page_content)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()