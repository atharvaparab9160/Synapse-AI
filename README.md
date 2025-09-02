# Synapse AI: A Community-Powered RAG Chatbot with Agentic Capabilities

Synapse AI is a full-stack, **Retrieval-Augmented Generation (RAG)** application designed to provide accurate, source-cited answers to user queries. It leverages a comprehensive knowledge base built by scraping thousands of real-world discussions from the Anaplan Community forums. This project demonstrates an end-to-end AI/ML workflow, from high-speed data collection and advanced retrieval strategies to a deployed, interactive web application with conversational memory.

[//]: # ([Live Demo]&#40;<!--- <<< 👈 PASTE YOUR LIVE STREAMLIT URL HERE --->&#41;)

---

## Key Features

- **High-Speed, Parallel Data Collection**  
  Engineered a robust web scraper using Python and `concurrent.futures` to extract over 14,000 Q&A threads, increasing data collection efficiency by over 10x compared to sequential methods.

- **Advanced RAG with Re-ranking**  
  Implements a two-stage retrieval process. After a fast initial search in a ChromaDB vector store, a sentence-transformers Cross-Encoder re-ranks the results for maximum contextual relevance, significantly improving answer quality.

- **Agentic Web Search & Guardrails**  
  The bot identifies "knowledge gaps" in its internal database. It uses an LLM-powered guardrail to check if the user's query is on-topic. If needed, it performs a live web search via the Tavily API for up-to-date information.

- **Conversational Memory**  
  Maintains full chat history, allowing users to ask natural, context-aware follow-up questions.

- **Professional UI & Deployment**  
  Polished, responsive user interface built with Streamlit, deployed on Streamlit Community Cloud, and connected to a hosted ChromaDB Cloud database for instant startup times.

---

## Tech Stack

- **Languages & Frameworks:** Python, LangChain, Streamlit, FastAPI  
- **AI & ML:** Hugging Face Transformers (`sentence-transformers`), Google Gemini 1.5 Flash, Tavily Search API  
- **Vector Database:** ChromaDB Cloud  
- **Data Handling:** BeautifulSoup, Requests, JSON, `concurrent.futures`  
- **Deployment & Tools:** Git, GitHub, Streamlit Community Cloud, Docker  

---

## Project Architecture

The system is decoupled into three main components:

1. **Data Pipeline**  
   - `scrap_with_all_comments.py`: Parallelized Python script that extracts and structures thousands of Q&A threads into JSON files.  
   - Includes retry mechanisms for handling large-scale scraping errors.

2. **Knowledge Base Creation**  
   - `create_vector_db_with_chunking.py`: Processes raw JSON, splits long documents into chunks, converts them into embeddings, and uploads to ChromaDB Cloud.

3. **RAG Application**  
   - `app.py`: Streamlit application that provides the UI and orchestrates the agentic workflow.  
   - Retrieves and re-ranks context, decides on live web search, and generates sourced answers via Gemini LLM.

---

## How to Run Locally

### 1. Clone the repository:
```bash
git clone https://github.com/atharvaparab9160/Synapse-AI.git
cd Synapse-AI
## Setup & Run Instructions
```
Follow these steps to set up and run Synapse AI locally:

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Your Credentials

Create a `.env` file in the root directory of the project and add your API keys:

```env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
CHROMA_API_KEY="YOUR_CHROMA_API_KEY"
CHROMA_TENANT="YOUR_CHROMA_TENANT_ID"
CHROMA_DATABASE="YOUR_CHROMA_DATABASE_NAME"
TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
```

> **Note:** Replace the placeholders with your actual API keys.

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the URL displayed in the terminal (usually `http://localhost:8501`) in your browser to access Synapse AI.

