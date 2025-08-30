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
# Import the CrossEncoder for re-ranking
from sentence_transformers.cross_encoder import CrossEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="Synapse AI",
    page_icon="✨",
    layout="wide"
)

# --- Custom CSS ---
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
    h1 {
        background: -webkit-linear-gradient(45deg, #4F8BF9, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .response-container {
        background-color: #1E222A;
        border-left: 5px solid #4F8BF9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        white-space: pre-wrap; /* Preserve formatting */
        word-wrap: break-word;
    }
    .response-container a { color: #63A6FF; text-decoration: none; }
    .response-container a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
COLLECTION_NAME = 'anaplan_community'


# --- Caching Functions ---
@st.cache_resource
def load_llm():
    """Loads the Language Model from Streamlit Secrets or .env file."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        st.error("🔴 Google API key not found. Please add it to your Streamlit Secrets or a .env file.")
        st.stop()

    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0,
                                  convert_system_message_to_human=True)


@st.cache_resource
def load_vector_store():
    """Connects to the hosted ChromaDB Cloud vector store."""
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")

    if not all([chroma_api_key, chroma_tenant, chroma_database]):
        load_dotenv()
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        chroma_tenant = os.getenv("CHROMA_TENANT")
        chroma_database = os.getenv("CHROMA_DATABASE")

    if not all([chroma_api_key, chroma_tenant, chroma_database]):
        st.error("🔴 ChromaDB credentials not found. Please add them to your Streamlit Secrets or a .env file.")
        st.stop()

    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    client = chromadb.CloudClient(
        tenant=chroma_tenant,
        database=chroma_database,
        api_key=chroma_api_key
    )

    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    return vectorstore


@st.cache_resource
def load_reranker():
    """Loads the Cross-Encoder model for re-ranking."""
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def format_docs(docs):
    """Prepares retrieved documents for the prompt."""
    if not docs: return "No relevant documents were found."
    return "\n\n".join(
        f"--- Source: {doc.metadata.get('title', 'N/A')} ---\nURL: {doc.metadata.get('url', 'N/A')}\nContent: {doc.page_content}"
        for doc in docs)


# --- Main App Logic ---
st.info("For any grievances or feedback, please contact us at: synapse.ai.help@gmail.com")
st.title("🚀 Synapse AI")
st.markdown("Your intelligent guide to community forums, powered by Gemini.")

# Load all necessary models and the vector store
llm = load_llm()
vectorstore = load_vector_store()
reranker = load_reranker()

# The retriever will fetch a larger number of initial documents for re-ranking
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

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

# --- User Interface ---
question = st.text_input("Ask a question:", placeholder="e.g., How do I use SUM with multiple conditions?")

if question:
    with st.spinner('Searching, analyzing, and re-ranking...'):
        try:
            # --- RAG with Re-ranking Workflow ---
            # 1. Retrieve a broad set of potentially relevant documents
            initial_docs = retriever.invoke(question)

            # 2. Re-rank these documents for higher relevance
            if initial_docs:
                # Create pairs of [question, document_text] for the Cross-Encoder
                pairs = [[question, doc.page_content] for doc in initial_docs]
                scores = reranker.predict(pairs)

                # Combine documents with their new scores and sort them
                doc_with_scores = list(zip(initial_docs, scores))
                doc_with_scores.sort(key=lambda x: x[1], reverse=True)

                # Select the top 5 documents after re-ranking
                reranked_docs = [doc for doc, score in doc_with_scores[:5]]

                # 3. Format only the best documents to be sent to the LLM
                context = format_docs(reranked_docs)
            else:
                context = "No relevant documents were found."

            # 4. Build and invoke the final chain with the high-quality, re-ranked context
            final_chain = prompt | llm | StrOutputParser()
            response = final_chain.invoke({"context": context, "question": question})

            # --- Display the response ---
            url_pattern = re.compile(r'https?://[^\s]+')
            formatted_response = url_pattern.sub(r'[\g<0>](\g<0>)', response)

            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="response-container">{formatted_response}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

