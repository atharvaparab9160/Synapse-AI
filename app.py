import os
import streamlit as st
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import re
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Page Configuration ---
st.set_page_config(
    page_title="Synapse AI",
    page_icon="🧠",
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
    # google_api_key = st.secrets.get("GOOGLE_API_KEY")
    # if not google_api_key:
    if True:
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
    # chroma_api_key = st.secrets.get("CHROMA_API_KEY")
    # chroma_tenant = st.secrets.get("CHROMA_TENANT")
    # chroma_database = st.secrets.get("CHROMA_DATABASE")
    #
    # if not all([chroma_api_key, chroma_tenant, chroma_database]):
    if True:
        load_dotenv()
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        chroma_tenant = os.getenv("CHROMA_TENANT")
        chroma_database = os.getenv("CHROMA_DATABASE")

    if not all([chroma_api_key, chroma_tenant, chroma_database]):
        st.error("🔴 ChromaDB credentials not found.")
        st.stop()

    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    client = chromadb.CloudClient(tenant=chroma_tenant, database=chroma_database, api_key=chroma_api_key)
    return Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=embedding_function)


@st.cache_resource
def load_reranker():
    """Loads the Cross-Encoder model for re-ranking."""
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


@st.cache_resource
def load_search_tool():
    """Loads the Tavily web search tool."""
    # tavily_api_key = st.secrets.get("TAVILY_API_KEY")
    # if not tavily_api_key:
    if True:
        load_dotenv()
        tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        st.error("🔴 Tavily API key not found.")
        st.stop()

    return TavilySearchResults(k=3, tavily_api_key=tavily_api_key)


def format_docs(docs):
    """Prepares retrieved documents for the prompt."""
    if not docs: return "No relevant documents were found."
    return "\n\n".join(
        f"--- Source: {doc.metadata.get('title', 'N/A')} ---\nURL: {doc.metadata.get('url', 'N/A')}\nContent: {doc.page_content}"
        for doc in docs)


# --- UI and Main App Logic ---
news_items = [
    "For any grievances or feedback, please contact us at: synapse.ai.help@gmail.com",
]
info_message = "  |  ".join(news_items)
st.info(info_message)
st.title("🚀 Synapse AI")
st.markdown("Your intelligent guide to community forums, powered by Gemini and live web search.")

# 1. Load all necessary components
llm = load_llm()
vectorstore = load_vector_store()
reranker = load_reranker()
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
web_search_tool = load_search_tool()

# 2. Define the Knowledge Gap Router Chain
gap_router_prompt_template = """Based on the user's question and the retrieved context, determine if the context is sufficient to provide a high-confidence answer. Respond with only 'Yes' or 'No'.
QUESTION: {question}
CONTEXT: {context}
"""
gap_router_prompt = ChatPromptTemplate.from_template(gap_router_prompt_template)
gap_router_chain = gap_router_prompt | llm | StrOutputParser()

# 3. NEW: Define the Topic Relevance Router Chain (Guardrail)
relevance_router_prompt_template = """You are a topic classifier. Your task is to determine if the user's question is related to Anaplan, business planning, finance, supply chain, or data modeling topics. Respond with only 'Yes' or 'No'.
QUESTION: {question}
"""
relevance_router_prompt = ChatPromptTemplate.from_template(relevance_router_prompt_template)
relevance_router_chain = relevance_router_prompt | llm | StrOutputParser()

# 4. Define the main Answer Generation Chain
answer_prompt_template = """
You are an expert Anaplan assistant. Your task is to answer the user's question based ONLY on the provided context.
Analyze the context provided below. It contains several sources, each with a URL.
Based on this context, synthesize a clear and concise answer (if needed, use bullet points).
If the context does not contain enough information to answer the question, state that you cannot find a specific answer in the provided sources. Do not make up information or use any external knowledge.
After your answer, you MUST list the URLs of all the sources you used to formulate your answer under a "Sources:" heading.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
answer_prompt = ChatPromptTemplate.from_template(answer_prompt_template)
answer_chain = answer_prompt | llm | StrOutputParser()


# 5. Define the Re-ranking function
def rerank_docs(inputs):
    question = inputs['question']
    docs = inputs['docs']
    if not docs:
        return []
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    doc_with_scores = list(zip(docs, scores))
    doc_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_with_scores[:5]]


# 6. Define the Agentic Workflow in a single function
def get_response(question: str):
    # Retrieve and re-rank documents from the internal knowledge base
    initial_docs = retriever.invoke(question)
    reranked_docs = rerank_docs({"question": question, "docs": initial_docs})
    formatted_context = format_docs(reranked_docs)

    # Check if the internal knowledge is sufficient
    decision = gap_router_chain.invoke({"question": question, "context": formatted_context})

    if "yes" in decision.lower():
        # If knowledge is sufficient, generate the answer directly
        return answer_chain.invoke({"question": question, "context": formatted_context})
    else:
        # If knowledge is insufficient, check if the topic is relevant
        st.info("Could not find a high-confidence answer in the knowledge base. Checking topic relevance...")
        is_relevant = relevance_router_chain.invoke({"question": question})

        if "no" in is_relevant.lower():
            # If the topic is not relevant, politely refuse to answer
            return "I am an assistant specialized in Anaplan and related topics. I cannot answer questions outside of this scope."
        else:
            # If the topic is relevant, perform a web search
            st.info("Topic is relevant. Searching the web for additional context...")
            web_results = web_search_tool.invoke(question)
            web_context = "\n\n".join(
                [f"--- Web Source ---\nURL: {result['url']}\nContent: {result['content']}" for result in web_results])

            # Combine internal and external context and generate the final answer
            combined_context = f"Internal Knowledge Base Context:\n{formatted_context}\n\nLive Web Search Context:\n{web_context}"
            return answer_chain.invoke({"question": question, "context": combined_context})


# --- User Interface ---
question = st.text_input("Ask a question:", placeholder="e.g., How do I use SUM with multiple conditions?")

if question:
    with st.spinner('Thinking... Performing retrieve, re-rank, and self-critique...'):
        try:
            response = get_response(question)
            url_pattern = re.compile(r'https?://[^\s)]+')
            formatted_response = url_pattern.sub(r'[\g<0>](\g<0>)', response)
            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="response-container">{formatted_response}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

