# 1. PROTOBUF FIX (Must be the absolute first thing in the file)
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 2. SQLITE FIX (For ChromaDB on Cloud)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 3. STANDARD IMPORTS
import streamlit as st
from dotenv import load_dotenv

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
                model="models/embedding-004", 
                google_api_key=gemini_api_key
            )
            
            print("DEBUG: Initializing Chroma Vector Database...")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            print("DEBUG: Assembling RAG chain components...")
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0)
            
            # (Keep your existing prompt and chain creation code here...)
            # ...
            
            st.session_state.conversational_rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )
            print("DEBUG: Pipeline successfully built and stored in session state!")
            
        st.success("✅ Conversational pipeline ready!")

# 4. Interactive Chat Interface
if "conversational_rag_chain" in st.session_state:
    
    # Display existing messages across app re-runs
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # Accept new conversational input
    user_input = st.chat_input("Ask something about your uploaded documents...")
    
    if user_input:
        # Instantly render user message
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.spinner("🤖 Thinking..."):
            # Execute pipeline passing the running history stream
            response = st.session_state.conversational_rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            
        # Render assistant response
        with st.chat_message("assistant"):
            st.write(answer)
            
        # Append latest turn back into the running history stream
        st.session_state.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer)
        ])
        
elif not uploaded_files:
    st.info("Please upload your PDFs to initiate the conversation.")