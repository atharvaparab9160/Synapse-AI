import os
import chromadb
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Anaplan Community AI Assistant",
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
DB_PATH = 'anaplan_db'
COLLECTION_NAME = 'anaplan_community'


# --- Caching Functions to improve performance ---
# @st.cache_resource
# def load_llm():
#     """Loads the Language Model, cached for performance."""
#     load_dotenv()
#     groq_api_key = os.getenv("GROQ_API_KEY")
#     if not groq_api_key:
#         st.error("🔴 Groq API key not found. Please create a .env file with GROQ_API_KEY.")
#         st.stop()
#     return ChatGroq(
#         groq_api_key=groq_api_key,
#         model_name="llama3-8b-8192",
#         temperature=0
#     )

@st.cache_resource
def load_llm():
    """Loads the Language Model, cached for performance."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("🔴 Google API key not found. Please create a .env file with GOOGLE_API_KEY.")
        st.stop()
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0,
        convert_system_message_to_human=True
    )


@st.cache_resource
def load_vector_store():
    """Loads the Vector Store and Retriever, cached for performance."""
    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    return vectorstore


def format_docs(docs):
    """Prepares the retrieved documents for insertion into the prompt."""
    if not docs:
        return "No relevant documents were found in the knowledge base."

    formatted_context = ""
    for i, doc in enumerate(docs):
        title = doc.metadata.get('title', 'No Title')
        url = doc.metadata.get('url', 'No URL')
        formatted_context += f"--- Source {i + 1}: {title} ---\n"
        formatted_context += f"URL: {url}\n"
        formatted_context += f"Content: {doc.page_content}\n\n"
    return formatted_context


# --- Main App Logic ---
st.title("🚀 Anaplan Synapse AI")
st.markdown(
    "Your intelligent guide to the Anaplan Community forums. Ask a question to get a synthesized answer with sources.")

# Load resources
llm = load_llm()
vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Define the prompt template
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

# Build the RAG chain
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# --- User Interface ---
question = st.text_input("Ask your Anaplan question:", placeholder="e.g., How do I use SUM with multiple conditions?")

if question:
    with st.spinner('Searching the Anaplan universe...'):
        try:
            response = rag_chain.invoke(question)

            # Use regex to find URLs and make them clickable
            url_pattern = re.compile(r'https?://[^\s]+')
            formatted_response = url_pattern.sub(r'[\g<0>](\g<0>)', response)

            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="response-container">{formatted_response}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

