# 1. PROTOBUF FIX (Must be the absolute first thing in the file)
import os
import time

# 1. PROTOBUF FIX (Must be the absolute first thing in the file)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 2. SQLITE FIX (For ChromaDB on Cloud)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # If we are running locally, just ignore this and use standard sqlite3

import streamlit as st
from dotenv import load_dotenv
# ... the rest of your imports continue normally below this

# Document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ... your other imports continue normally below this
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LLM, Memory, and Chain assemblies
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load local environment variables
load_dotenv()

st.set_page_config(page_title="Conversational PDF Assistant", page_icon="💬", layout="centered")
st.title("💬 Conversational Multi-PDF Assistant")
st.write("Upload documents and have a continuous chat about their contents.")

# Initialize persistent chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 1. Sidebar API Key Fallback
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    gemini_api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

# 2. Document Upload Interface
uploaded_files = st.file_uploader("Upload 3 to 5 PDFs", type="pdf", accept_multiple_files=True)

# 3. Process Documents and Construct the Conversational RAG Pipeline
if uploaded_files:
    print(f"DEBUG: Files uploaded count = {len(uploaded_files)}")
    print(f"DEBUG: API Key present = {bool(gemini_api_key)}")
    
    if not gemini_api_key:
        st.warning("⚠️ Files received, but Gemini API Key is missing. Please check your configuration.")
    
    if gemini_api_key and "conversational_rag_chain" not in st.session_state:
        with st.spinner("🔄 Indexing documents locally..."):
            print("DEBUG: Starting document parsing...")
            all_docs = []
            
            for uploaded_file in uploaded_files:
                print(f"DEBUG: Reading file: {uploaded_file.name}")
                temp_filename = f"temp_{uploaded_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(temp_filename)
                all_docs.extend(loader.load())
                os.remove(temp_filename)
            
            print(f"DEBUG: Successfully split into {len(all_docs)} raw pages. Starting chunking...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            
            print("DEBUG: Generating cloud embeddings via Google GenAI...")
            embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=gemini_api_key
            )
            
            print("DEBUG: Initializing Chroma Vector Database...")
            vectorstore = Chroma(embedding_function=embeddings)
            batch_size = 90
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            print(f"DEBUG: Processing chunks {i} to {i + len(batch)} of {len(splits)}...")
            vectorstore.add_documents(batch)
    
    # If there are more chunks left, sleep for 60 seconds to reset the Google limit
            if i + batch_size < len(splits):
                print("DEBUG: Pausing for 60 seconds to avoid API rate limits...")
                with st.spinner("Pausing for 60s to respect Google's free tier limits..."):
                    time.sleep(61)

            retriever = vectorstore.as_retriever()
            
            print("DEBUG: Assembling RAG chain components...")
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0)
            
            # --- START OF RESTORED CODE ---
            # Sub-Chain A: History-Aware Query Condenser Prompt
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
            
            # Sub-Chain B: Core QA Generation Prompt
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Keep the answer concise.\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            # --- END OF RESTORED CODE ---
            
            # Final Assembly
            st.session_state.conversational_rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )
            print("DEBUG: Pipeline successfully built and stored in session state!")
            
        st.success("✅ Conversational pipeline ready!")