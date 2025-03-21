import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import streamlit as st

logging.basicConfig(level=logging.INFO)

DOC_PATH = r"C:/Users/ankam/Documents/password-reset.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_vector_db(chunks):
    """Create a vector database from document chunks."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant with two roles:
                    If the user's question is a general conversational query (e.g., greetings, casual talk, time, weather), respond naturally and conversationally.
                    If the user's question pertains to specific details in a document, your task is to generate five different alternative versions of the user's original question. 
                    These variations will be used to retrieve relevant documents from a vector database, helping the user overcome limitations of the distance-based similarity search.
                    For general questions, answer directly and conversationally.
                    For document-related questions, provide five alternative queries clearly separated by newlines.
                    Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created successfully.")
    return chain


def main():
    st.title("📚 Welcome to your AI Document Assistant")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Load PDF and create embeddings only once
    if "vector_db" not in st.session_state:
        with st.spinner("Loading and processing PDF..."):
            data = ingest_pdf(DOC_PATH)
            if data is None:
                st.error("Could not load document.")
                return
            chunks = split_documents(data)
            st.session_state.vector_db = Chroma.from_documents(
                chunks,
                embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
                collection_name=VECTOR_STORE_NAME
            )
            st.session_state.llm = ChatOllama(model=MODEL_NAME)
            st.session_state.retriever = create_retriever(st.session_state.vector_db, st.session_state.llm)
            st.session_state.chain = create_chain(st.session_state.retriever, st.session_state.llm)

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])

    # Handle new user input
    user_input = st.chat_input("How can I help you?")
    if user_input:
        # Display user's question immediately
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    res = st.session_state.chain.invoke(user_input)
                    st.write(res)
                    st.session_state.messages.append({"role": "assistant", "content": res})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()