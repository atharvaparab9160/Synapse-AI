import os
import streamlit as st
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Synapse AI",
    page_icon="✨",
    layout="wide"
)

# --- Custom CSS for "Out of this World" UI ---
st.markdown("""
<style>
    /* General Styling */
    .stApp {
        background-color: #0E1117; /* Dark background */
        color: #FAFAFA;
    }
    .stTextInput > div > div > input {
        background-color: #1E222A;
        color: #FAFAFA;
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #4F8BF9;
        background-color: transparent;
        color: #4F8BF9;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #4F8BF9;
        color: white;
        border-color: #4F8BF9;
    }

    /* Title with Gradient */
    h1 {
        background: -webkit-linear-gradient(45deg, #4F8BF9, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }

    /* Response Box Styling */
    .response-container {
        background-color: #1E222A;
        border-left: 5px solid #4F8BF9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        white-space: pre-wrap; /* Preserve formatting */
        word-wrap: break-word;
    }
    .response-container h3 {
        color: #FAFAFA;
        margin-top: 0;
    }
    .response-container a {
        color: #63A6FF;
        text-decoration: none;
    }
    .response-container a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
COLLECTION_NAME = 'anaplan_community'

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
# --- Caching Functions ---
@st.cache_resource
def load_llm():
    """Loads the Language Model from Streamlit Secrets or .env file."""
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
<<<<<<< Updated upstream
    
    if not google_api_key:
        st.error("🔴 Google API key not found. Please add it to your Streamlit Secrets or a .env file.")
        st.stop()
        
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0, convert_system_message_to_human=True)
=======

    if not google_api_key:
        st.error("🔴 Google API key not found. Please add it to your Streamlit Secrets or a .env file.")
        st.stop()

    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0,
                                  convert_system_message_to_human=True)

>>>>>>> Stashed changes

@st.cache_resource
def load_vector_store():
    """Connects to the hosted ChromaDB Cloud vector store."""
    chroma_api_key = st.secrets.get("CHROMA_API_KEY")
    chroma_tenant = st.secrets.get("CHROMA_TENANT")
    chroma_database = st.secrets.get("CHROMA_DATABASE")

    if not all([chroma_api_key, chroma_tenant, chroma_database]):
        load_dotenv()
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        chroma_tenant = os.getenv("CHROMA_TENANT")
        chroma_database = os.getenv("CHROMA_DATABASE")

    if not all([chroma_api_key, chroma_tenant, chroma_database]):
        st.error("🔴 ChromaDB credentials not found. Please add them to your Streamlit Secrets or a .env file.")
        st.stop()

    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
<<<<<<< Updated upstream
    
=======

>>>>>>> Stashed changes
    client = chromadb.CloudClient(
        tenant=chroma_tenant,
        database=chroma_database,
        api_key=chroma_api_key
    )
<<<<<<< Updated upstream
    
=======

>>>>>>> Stashed changes
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    return vectorstore

def format_docs(docs):
    """Prepares retrieved documents for the prompt."""
    if not docs: return "No relevant documents were found."
<<<<<<< Updated upstream
    return "\n\n".join(f"--- Source: {doc.metadata.get('title', 'N/A')} ---\nURL: {doc.metadata.get('url', 'N/A')}\nContent: {doc.page_content}" for doc in docs)
=======
    return "\n\n".join(
        f"--- Source: {doc.metadata.get('title', 'N/A')} ---\nURL: {doc.metadata.get('url', 'N/A')}\nContent: {doc.page_content}"
        for doc in docs)

>>>>>>> Stashed changes

# --- Main App Logic ---
st.title("🚀 Synapse AI")
st.markdown("Your intelligent guide to community forums, powered by Gemini.")

llm = load_llm()
vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

template = """
You are an expert Anaplan assistant. Your task is to answer the user's question based ONLY on the provided context.
Analyze the context provided below. It contains several sources, each with a URL.
Based on this context, synthesize a clear and concise answer (if needed answer in proper points).
If the context does not contain enough information to answer the question, state that you cannot find a specific answer in the provided sources. Do not make up information or use any external knowledge.
answer in proper pointe rformat ,Make sure the response looks good when rendered in Streamlit’s st.markdown, without large gaps or misaligned text.
After your answer, you MUST list the URLs of all the sources you used to formulate your answer under a "Sources:" heading.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

question = st.text_input("Ask a question:", placeholder="e.g., How do I use SUM with multiple conditions?")

if question:
    with st.spinner('Searching the knowledge base...'):
        try:
            response = rag_chain.invoke(question)
            url_pattern = re.compile(r'https?://[^\s]+')
            formatted_response = url_pattern.sub(r'[\g<0>](\g<0>)', response)
            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="response-container">{formatted_response}</div>', unsafe_allow_html=True)
        except Exception as e:
<<<<<<< Updated upstream
            st.error(f"An error occurred: {e}")
=======
            st.error(f"An error occurred: {e}")
>>>>>>> Stashed changes
