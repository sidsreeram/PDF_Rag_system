import os
import streamlit as st
from dotenv import load_dotenv

# Document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
if uploaded_files and gemini_api_key:
    
    if "conversational_rag_chain" not in st.session_state:
        with st.spinner("🔄 Indexing documents locally with Hugging Face embeddings..."):
            all_docs = []
            
            for uploaded_file in uploaded_files:
                temp_filename = f"temp_{uploaded_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(temp_filename)
                all_docs.extend(loader.load())
                os.remove(temp_filename)
            
            # Chunking and local vector db generation
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=gemini_api_key
            )
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            # Initialize LLM
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0)
            
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
            
            # Create the sub-chain that condenses vague queries using history
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
            
            # Assemble the complete conversational pipeline
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            st.session_state.conversational_rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )
            
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