import logging
import os
import shutil
import sys
from typing import List

import openai
import streamlit as st

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    GitLoader,
    NotebookLoader,
    OnlinePDFLoader,
    PythonLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

from constants import (
    APP_NAME,
    CHUNK_SIZE,
    DATA_PATH,
    FETCH_K,
    MAX_TOKENS,
    MODEL,
    PAGE_ICON,
    TEMPERATURE,
    K,
)


# configure logger
logger = logging.getLogger(APP_NAME)

def configure_logger(debug: int = 0) -> None:
    log_level = logging.DEBUG if debug == 1 else logging.INFO
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)

    formatter = logging.Formatter("%(message)s")

    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.propagate = False

configure_logger(0)


# authenticaiton logic
def authenticate(
    openai_api_key: str
) -> None:
    
    openai_api_key = (
        openai_api_key
        or os.environ.get("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY")
    )
    if not (openai_api_key):
        st.session_state["auth_ok"] = False
        st.error("Correct API Keys not Found!", icon=PAGE_ICON)
        return
    try:
        # Try to access openai and deeplake
        with st.spinner("Authenticating..."):
            openai.api_key = openai_api_key
            openai.Model.list()
    except Exception as e:
        logger.error(f"Authentication failed error: {e}")
        st.session_state["auth_ok"] = False
        st.error("Authentication failed", icon=PAGE_ICON)
        return
    # store credentials in the session state
    st.session_state["auth_ok"] = True
    st.session_state["openai_api_key"] = openai_api_key
    logger.info("Authentification successful!")

# file upload and save logic
def save_uploaded_file(uploaded_file: UploadedFile) -> str:
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    file_path = str(DATA_PATH / uploaded_file.name)
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    file = open(file_path, "wb")
    file.write(file_bytes)
    file.close()
    logger.info(f"Uploaded File Saved: {file_path}")
    return file_path

# file delete logic
def delete_uploaded_file(uploaded_file: UploadedFile) -> None:
    file_path = DATA_PATH / uploaded_file.name
    if os.path.exists(DATA_PATH):
        os.remove(file_path)
        logger.info(f"Uploaded File Removed: {file_path}")

# file load error logic
def handle_load_error(e: str = None) -> None:
    error_msg = f"Loading Error '{st.session_state['data_source']}':\n\n{e}"
    st.error(error_msg, icon=PAGE_ICON)
    logger.error(error_msg)
    st.stop()

# special logic for getting git repo data
def load_git(data_source: str, chunk_size: int = CHUNK_SIZE) -> List[Document]:
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    repo_name = data_source.split("/")[-1].split(".")[0]
    repo_path = str(DATA_PATH / repo_name)
    clone_url = data_source
    if os.path.exists(repo_path):
        clone_url = None
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
    )
    branches = ["main", "master"]
    for branch in branches:
        try:
            docs = GitLoader(repo_path, clone_url, branch).load_and_split(text_splitter)
            break
        except Exception as e:
            logger.error(f"Error loading git repo: {e}")
    if os.path.exists(repo_path):
        # cleanup repo afterwards
        shutil.rmtree(repo_path)
    try:
        return docs
    except:
        msg = "Make sure to use HTTPS based git repo links"
        handle_load_error(msg)

# swiss army knife of all file types which are supported by langchain
def load_any_data_source(
    data_source: str, chunk_size: int = CHUNK_SIZE
) -> List[Document]:
    # langchain baed multiple file loading logic
    is_dir = os.path.isdir(data_source)
    is_file = os.path.isfile(data_source)
    is_text = data_source.endswith(".txt")
    is_web = data_source.startswith("http")
    is_pdf = data_source.endswith(".pdf")
    is_csv = data_source.endswith("csv")
    is_html = data_source.endswith(".html")
    is_git = data_source.endswith(".git")
    is_notebook = data_source.endswith(".ipynb")
    is_doc = data_source.endswith(".doc")
    is_py = data_source.endswith(".py")

    loader = None
    if is_dir:
        loader = DirectoryLoader(data_source, recursive=True, silent_errors=True)
    elif is_git:
        return load_git(data_source, chunk_size)
    elif is_web:
        if is_pdf:
            loader = OnlinePDFLoader(data_source)
        else:
            loader = WebBaseLoader(data_source)
    elif is_file:
        if is_text:
            loader = TextLoader(data_source)
        elif is_notebook:
            loader = NotebookLoader(data_source)
        elif is_pdf:
            loader = UnstructuredPDFLoader(data_source)
        elif is_html:
            loader = UnstructuredHTMLLoader(data_source)
        elif is_doc:
            loader = UnstructuredWordDocumentLoader(data_source)
        elif is_csv:
            loader = CSVLoader(data_source, encoding="utf-8")
        elif is_py:
            loader = PythonLoader(data_source)
        else:
            loader = UnstructuredFileLoader(data_source)
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0
        )
        docs = loader.load_and_split(text_splitter)
        return docs
    except Exception as e:
        msg = (
            e
            if loader
            else f"FileQueryBot doesn't support your file format as of now!"
        )
        handle_load_error(msg)

# vector db logic using chroma now but can be changed to any supported vector db by langchain
def setup_vector_store(data_source: str, chunk_size: int = CHUNK_SIZE) -> VectorStore:
    embeddings = OpenAIEmbeddings(
        disallowed_special=(), openai_api_key=st.session_state["openai_api_key"]
    )
    docs = load_any_data_source(data_source, chunk_size)
    vector_store = None
    with st.spinner("Loading data into vector store..."):
            vector_store = Chroma.from_documents(docs, embeddings)
    return vector_store

# main langhcain setup logic
def build_chain(
    data_source: str,
    k: int = K,
    fetch_k: int = FETCH_K,
    chunk_size: int = CHUNK_SIZE,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> ConversationalRetrievalChain:
   
    # build ConversationalRetievalChain using model
    vector_store = setup_vector_store(data_source, chunk_size)
    retriever = vector_store.as_retriever()
    search_kwargs = {
        "maximal_marginal_relevance": True,
        "distance_metric": "cos",
        "fetch_k": fetch_k,
        "k": k,
    }
    retriever.search_kwargs.update(search_kwargs)
    model = ChatOpenAI(
        model_name=MODEL,
        temperature=temperature,
        openai_api_key=st.session_state["openai_api_key"],
    )
    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        chain_type="stuff",
        verbose=True,
        # gpt 3-5 turbo model token limit => 4096
        max_tokens_limit=max_tokens,
    )
    return chain

# utility method to use build_chain method, save in session & use it in UI
def update_chain() -> None:
    try:
        st.session_state["chain"] = build_chain(
            data_source=st.session_state["data_source"],
            k=st.session_state["k"],
            fetch_k=st.session_state["fetch_k"],
            chunk_size=st.session_state["chunk_size"],
            temperature=st.session_state["temperature"],
            max_tokens=st.session_state["max_tokens"],
        )
        st.session_state["chat_history"] = []
        st.success("All Set, Let's Chat Now!", icon="✅")
    except Exception as e:
        msg = f"Error: {e} building chain from '{st.session_state['data_source']}'"
        logger.error(msg)
        st.error(msg, icon=PAGE_ICON)

# utility method for showing content in UI
def generate_response(prompt: str) -> str:
    with st.spinner("Generating response"), get_openai_callback() as cb:
        response = st.session_state["chain"](
            {"question": prompt, "chat_history": st.session_state["chat_history"]}
        )
    st.session_state["chat_history"].append((prompt, response["answer"]))
    return response["answer"]
