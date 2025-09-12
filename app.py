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
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
# --- NEW IMPORT FOR THE SPECIFIC RATE LIMIT ERROR ---
from google.api_core import exceptions as google_exceptions

# --- Page Configuration (UNCHANGED) ---
st.set_page_config(
    page_title="Synapse AI",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS (UNCHANGED) ---
st.markdown("""
<style>
    /* Main App Styling */
    .stApp {
        background-color: #0E1117; /* Dark background */
        color: #FAFAFA;
    }
    .main .block-container {
        padding-top: 5rem;   /* ⬅️ more padding so navbar doesn’t overlap */
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1E222A;
        border-right: 1px solid #262730;
    }
    [data-testid="stSidebar"] h1 { /* Sidebar Title */
        color: #FAFAFA;
        font-weight: bold;
        font-size: 2.5rem;
        background: -webkit-linear-gradient(45deg, #4F8BF9, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton > button { /* Sidebar & Suggested Prompts Buttons */
        border-radius: 8px;
        border: 1px solid #4A4A4A;
        background-color: transparent;
        color: #D0D0D0;
        text-align: left;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
        margin-bottom: 0.5rem;
    }
    .stButton > button:hover {
        background-color: #262730;
        color: #FFFFFF;
        border-color: #4F8BF9;
    }
    /* Main Chat Area Styling */
    [data-testid="stChatMessage"] {
        border-radius: 20px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        border: none !important;
        display: inline-block !important;
        width: auto !important;
        max-width: 70% !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }
    [data-testid="stChatMessageUser"] {
        background-color: #4F8BF9;
        margin-left: auto;
        color: white;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: #262730;
        color: #FAFAFA;
    }
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] div { margin: 0; line-height: 1.4; }
    [data-testid="stChatMessage"] { display: flex !important; align-items: flex-start !important; }
    [data-testid="stChatMessage"] img { margin-top: 0 !important; }
    .stChatInput { background-color: #0E1117; }
    /* Hiding Streamlit Branding */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# --- Caching Functions (load_llm is REMOVED as it's now dynamic) ---
@st.cache_resource
def load_vector_store():
    load_dotenv()
    chroma_api_key = os.getenv("CHROMA_API_KEY") or st.secrets.get("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT") or st.secrets.get("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE") or st.secrets.get("CHROMA_DATABASE")
    if not all([chroma_api_key, chroma_tenant, chroma_database]): st.error(
        "🔴 ChromaDB credentials not found."); st.stop()
    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    client = chromadb.CloudClient(tenant=chroma_tenant, database=chroma_database, api_key=chroma_api_key)
    return Chroma(client=client, collection_name='anaplan_community', embedding_function=embedding_function)


@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


@st.cache_resource
def load_search_tool():
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")
    if not tavily_api_key: st.error("🔴 Tavily API key not found."); st.stop()
    return TavilySearchResults(k=3, tavily_api_key=tavily_api_key)


# --- Helper Functions ---
def format_docs(docs):
    if not docs: return "No relevant documents were found."
    return "\n\n".join(
        f"--- Source: {doc.metadata.get('title', 'N/A')} ---\nURL: {doc.metadata.get('url', 'N/A')}\nContent: {doc.page_content}"
        for doc in docs), [doc.metadata.get('url', 'N/A') for doc in docs]


def format_chat_history_for_chain(messages):
    return [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg
            in messages]


def rerank_docs(inputs, reranker_model):
    question = inputs['question']
    docs = inputs['docs']
    if not docs: return []
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker_model.predict(pairs)
    doc_with_scores = list(zip(docs, scores))
    doc_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_with_scores[:5]]


# --- RE-ARCHITECTED: Full Agentic Workflow with Self-Healing Key Management ---
def get_response(user_question: str, chat_history: list):
    load_dotenv()

    # 1. Load the pool of API keys from numbered environment variables
    api_keys = []

    key = os.getenv(f"GOOGLE_API_KEYS") or st.secrets.get(f"GOOGLE_API_KEYS")
    if key:
        api_keys = list(key.split(","))

    if not api_keys:
        return "Error: No Google API keys found. Please add GOOGLE_API_KEY_1, etc. to your secrets."

    # Initialize key statuses in session state if they don't exist
    if 'key_statuses' not in st.session_state:
        st.session_state.key_statuses = {f"Key {i + 1}": "✅ Active" for i in range(len(api_keys))}
    if 'key_index' not in st.session_state:
        st.session_state.key_index = 0

    # 2. Loop through the keys, attempting to get a response
    start_index = st.session_state.key_index

    for i in range(len(api_keys)):
        current_index = (start_index + i) % len(api_keys)
        key_name = f"Key {current_index + 1}"

        if st.session_state.key_statuses.get(key_name) == "❌ Exhausted":
            continue

        try:
            current_key = api_keys[current_index]
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=current_key, temperature=0,
                                         convert_system_message_to_human=True)

            # --- The entire RAG and Agentic pipeline now runs inside this loop ---

            # A. Create standalone question
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 '''You are an expert search query reformulator for Anaplan. Your task is to create a single, self-contained search query based on the conversation history and the user's follow-up question for document retrieval.
                 Follow these rules:
                 1.) If the follow-up question is a direct continuation or clarification of the conversation history, merge them to form a complete query that fully reflects the user's intent without adding unnecessary words.
                 2.) If the follow-up question is about a new topic or unrelated to the previous conversation, IGNORE the history and create the query only from the new question.
                 3.) If the conversation history is empty, create the query using only the follow-up question.
                 4.) Do not add words that are not present or strongly implied in the conversation. Stick to keywords from the user's input.
                 5.) Ensure the final query is brief, precise, and suitable for retrieving relevant documents. Avoid filler words, generic phrases, or assumptions.
                 STRICTLY REMEMBER:  
                 - Output ONLY the final search query—no explanations or extra text.  
                 - Do NOT answer the question or elaborate; only generate the query for document retrieval.  
                 - Do not invent or infer terms unless clearly implied by the conversation context.
                 '''
                 ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            standalone_question = (contextualize_q_prompt | llm | StrOutputParser()).invoke({
                "chat_history": chat_history, "input": user_question
            })

            # B. Retrieve and Re-rank
            retriever = load_vector_store().as_retriever(search_kwargs={"k": 20})
            reranker = load_reranker()
            # st.write("standalone :",standalone_question)
            retrieved_docs = retriever.invoke(standalone_question)
            reranked_docs = rerank_docs({"question": standalone_question, "docs": retrieved_docs}, reranker)
            formatted_context, links = format_docs(reranked_docs)
            # st.write(links)

            # C. Agentic Routing
            gap_router_prompt = ChatPromptTemplate.from_template("""Based on the user's question, and the retrieved context, determine if the context is sufficient to provide a high-confidence answer. Respond with only 'Yes' or 'No'.
            QUESTION: {question}
            CONTEXT: {context}""")
            gap_router_chain = gap_router_prompt | llm | StrOutputParser()
            decision = gap_router_chain.invoke({"question": standalone_question, "context": formatted_context})

            # D. Final Answer Generation
            answer_prompt = ChatPromptTemplate.from_template("""
            You are an expert Anaplan assistant.  
            Use all the information provided in the context to create a comprehensive and accurate answer to the user's question.  
            - Base your answer strictly on the provided context. Do not include information from outside the context.  
            - Combine all relevant details from the context to ensure the answer is as complete and helpful as possible.  
            - Present the answer as clear, concise, and professional bullet points.  
            - If the context does not provide sufficient information to answer the question confidently, explicitly state that no specific answer is available and do not speculate or add outside knowledge.  
            - After the answer, include a "Sources:" section listing all relevant source URLs mentioned in the context. If no sources are provided, write "Sources: None".  
            CONTEXT:  
            {context}  

            QUESTION:  
            {question}  

            ANSWER:

            """)
            answer_chain = answer_prompt | llm | StrOutputParser()

            if "yes" in decision.lower():
                response = answer_chain.invoke({"question": standalone_question, "context": formatted_context})
            else:
                relevance_router_prompt = ChatPromptTemplate.from_template("""
                You are a topic classifier. Your task is to determine if the user's question is related to Anaplan, business planning, finance, supply chain, or data modeling topics. 
                Even if it is little realted with anaplan try responding with 'Yes'
                Respond with only 'Yes' or 'No'.
                QUESTION: {question}
                """)
                relevance_router_chain = relevance_router_prompt | llm | StrOutputParser()
                is_relevant = relevance_router_chain.invoke({"question": standalone_question})

                if "no" in is_relevant.lower():
                    response = "I am an assistant specialized in Anaplan and related topics. I cannot answer questions outside of this scope."
                else:
                    web_search_tool = load_search_tool()
                    web_results = web_search_tool.invoke(standalone_question)
                    web_context = "\n\n".join(
                        [f"--- Web Source ---\nURL: {result['url']}\nContent: {result['content']}" for result in
                         web_results if isinstance(result, dict)])
                    combined_context = f"Internal Knowledge Base Context:\n{formatted_context}\n\nLive Web Search Context:\n{web_context}"
                    response = answer_chain.invoke({"question": standalone_question, "context": combined_context})

            st.session_state.key_index = (current_index + 1) % len(api_keys)
            return response

        except Exception as e:
            st.toast(f"Current API Key is exhausted. Trying the next available key...", icon="🔑")
            # st.warning(f"Current API Key is exhausted. Trying the next available key...", icon="🔑")
            st.session_state.key_statuses[key_name] = "❌ Exhausted"
            continue

        # except Exception as e:
        #     return f"An unexpected error occurred with {key_name}: {e}"

    return "Error: All available API keys have reached their daily limit. Please try again tomorrow."


# --- UI Rendering ---
with st.sidebar:
    st.title("🚀 Synapse AI")
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.key_index = 0
        st.session_state.key_statuses = {}
        st.rerun()
    st.markdown("### Relevant Prompts")
    suggested_prompts = ["What is Selective Access?", "How do I use SUM Function?", "Tell me about ALM.",
                         "Best Practices for multi-select filter."]


    def handle_sidebar_click(prompt):
        st.session_state.clicked_prompt = prompt


    for prompt_text in suggested_prompts:
        st.button(prompt_text, on_click=handle_sidebar_click, args=[prompt_text], use_container_width=True)
    st.info("For any grievances or feedback, please contact us at: synapse.ai.help@gmail.com")

    st.markdown("---")
    st.markdown("### API Key Status")
    if 'key_statuses' in st.session_state and st.session_state.key_statuses:
        for key, status in st.session_state.key_statuses.items():
            st.markdown(f"- {key}: {status}")
    else:
        st.markdown("No keys loaded yet.")

st.header("New Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]


def process_and_display(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chat_history_for_chain = format_chat_history_for_chain(st.session_state.messages[:-1])
                response = get_response(prompt, chat_history_for_chain)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.get("clicked_prompt"):
    prompt_to_run = st.session_state.clicked_prompt
    st.session_state.clicked_prompt = None
    process_and_display(prompt_to_run)
    st.rerun()

if prompt := st.chat_input("Ask your question..."):
    process_and_display(prompt)