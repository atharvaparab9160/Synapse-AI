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
# --- NEW IMPORTS FOR MEMORY ---
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Synapse AI",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS for Professional Chat UI (UNCHANGED) ---
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
    .st-emotion-cache-16txtl3 { /* Sidebar Title */
        color: #FAFAFA;
        font-weight: bold;
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

# --- Configuration & Caching Functions (UNCHANGED) ---
COLLECTION_NAME = 'anaplan_community'


@st.cache_resource
def load_llm():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("🔴 Google API key not found.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0,
                                  convert_system_message_to_human=True)


@st.cache_resource
def load_vector_store():
    load_dotenv()
    chroma_api_key = os.getenv("CHROMA_API_KEY") or st.secrets.get("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT") or st.secrets.get("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE") or st.secrets.get("CHROMA_DATABASE")
    if not all([chroma_api_key, chroma_tenant, chroma_database]):
        st.error("🔴 ChromaDB credentials not found.")
        st.stop()
    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    client = chromadb.CloudClient(tenant=chroma_tenant, database=chroma_database, api_key=chroma_api_key)
    return Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=embedding_function)


@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


@st.cache_resource
def load_search_tool():
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")
    if not tavily_api_key:
        st.error("🔴 Tavily API key not found.")
        st.stop()
    return TavilySearchResults(k=3, tavily_api_key=tavily_api_key)


# --- Helper Functions ---
def format_docs(docs):
    if not docs: return "No relevant documents were found."
    return "\n\n".join(
        f"--- Source: {doc.metadata.get('title', 'N/A')} ---\nURL: {doc.metadata.get('url', 'N/A')}\nContent: {doc.page_content}"
        for doc in docs)


def format_chat_history_for_chain(messages):
    return [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg
            in messages]


# --- Main App Logic & RAG Chains ---
llm = load_llm()
vectorstore = load_vector_store()
reranker = load_reranker()
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
web_search_tool = load_search_tool()

# --- NEW: History-Aware Rewriting Chain ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. If the question is not related with the History return the question as it is by just refining the prompt. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever_chain = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# --- Agentic Chains (Gap and Relevance) ---
gap_router_prompt = ChatPromptTemplate.from_template("""Based on the user's question, chat history, and the retrieved context, determine if the context is sufficient to provide a high-confidence answer. Respond with only 'Yes' or 'No'.
QUESTION: {question}
CONTEXT: {context}""")
gap_router_chain = gap_router_prompt | llm | StrOutputParser()

relevance_router_prompt = ChatPromptTemplate.from_template("""
You are a topic classifier. Your task is to determine if the user's question is related to Anaplan, business planning, finance, supply chain, or data modeling topics. 
Even if it is little realted with anaplan try responding with 'Yes'
Respond with only 'Yes' or 'No'.
QUESTION: {question}
""")
relevance_router_chain = relevance_router_prompt | llm | StrOutputParser()

# --- CORRECTED: Final Answer Generation Chain (NO LONGER takes chat_history) ---
answer_prompt = ChatPromptTemplate.from_template("""You are an expert Anaplan assistant.  
Answer the user's question **strictly using ONLY the information in the provided context**.  
- Present your answer in **clear bullet points**.  
- Keep each point concise and professional.  
- If the context does not contain enough information to answer confidently, explicitly state that no specific answer is available in the provided sources. Do not infer or fabricate information.  
- After the answer, include a "Sources:" section, with each source URL listed as a bullet point. If no sources are available, write "Sources: None".  

CONTEXT:  
{context}  

QUESTION:  
{question}  

ANSWER:
""")
answer_chain = answer_prompt | llm | StrOutputParser()


def rerank_docs(inputs):
    question = inputs['question']
    docs = inputs['docs']
    if not docs: return []
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    doc_with_scores = list(zip(docs, scores))
    doc_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_with_scores[:5]]


# --- UPDATED: Full agentic workflow with CONVERSATIONAL MEMORY ---
def get_response(question: str, chat_history: list):
    # 1. First, create a standalone question using the history
    standalone_question_chain = contextualize_q_prompt | llm | StrOutputParser()
    standalone_question = standalone_question_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })

    # 2. Retrieve documents based on the standalone question
    # st.title(standalone_question)
    retrieved_docs = retriever.invoke(standalone_question)

    # 3. Re-rank the retrieved documents
    reranked_docs = rerank_docs({"question": standalone_question, "docs": retrieved_docs})
    formatted_context = format_docs(reranked_docs)
    # st.write(formatted_context)

    # 4. Agentic "Knowledge Gap" Check (uses the standalone question)
    decision = gap_router_chain.invoke({
        "question": standalone_question,
        "context": formatted_context
    })

    if "yes" in decision.lower():
        # 5a. Generate answer from internal knowledge
        return answer_chain.invoke({
            "question": standalone_question,
            "context": formatted_context
        })
    else:
        # 5b. Agentic "Relevance Guardrail" Check
        is_relevant = relevance_router_chain.invoke({"question": standalone_question})
        if "no" in is_relevant.lower():
            return "I am an assistant specialized in Anaplan and related topics. I cannot answer questions outside of this scope."
        else:
            # 5c. Perform Web Search
            web_results = web_search_tool.invoke(standalone_question)
            web_context = "\n\n".join(
                [f"--- Web Source ---\nURL: {result['url']}\nContent: {result['content']}" for result in web_results])
            combined_context = f"Internal Knowledge Base Context:\n{formatted_context}\n\nLive Web Search Context:\n{web_context}"
            # 5d. Generate final answer with combined context
            return answer_chain.invoke({
                "question": standalone_question,
                "context": combined_context
            })


# --- UI Rendering (UNCHANGED) ---
with st.sidebar:
    st.title("Synapse AI")
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### Relevant Prompts")
    suggested_prompts = ["What is Selective Access?", "How do I use SUM Function?", "Tell me about ALM.",
                         "Best Practices for multi-select filter."]


    def handle_sidebar_click(prompt):
        st.session_state.clicked_prompt = prompt


    for prompt_text in suggested_prompts:
        st.button(prompt_text, on_click=handle_sidebar_click, args=[prompt_text], use_container_width=True)

    st.info("For any grievances or feedback, please contact us at: synapse.ai.help@gmail.com")

st.header("New Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with your Anaplan questions today?"}]


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




