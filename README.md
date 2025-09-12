# Synapse AI - Anaplan Community Assistant

[//]: # ([]&#40;https://www.python.org/downloads/&#41;  )

[//]: # ([]&#40;https://www.langchain.com/&#41;  )

[//]: # ([]&#40;https://streamlit.io/&#41;  )

[//]: # ([]&#40;https://www.google.com/search?q=YOUR_STREAMLIT_APP_URL_HERE&#41;  )

[//]: # ()
[//]: # (**[▶️ View the Live Demo]&#40;https://www.google.com/search?q=YOUR_STREAMLIT_APP_URL_HERE&#41;**)

Synapse AI is an advanced, agentic Retrieval-Augmented Generation (RAG) chatbot designed to be a subject-matter expert on the Anaplan platform. It leverages a comprehensive knowledge base built from over 14,000 real-world discussion threads from the Anaplan Community forum to provide accurate, context-aware, and source-cited answers to user questions.

---

## Key Features & Engineering Highlights

This project goes beyond a simple RAG implementation by incorporating a full engineering lifecycle, from data acquisition to a deployed, resilient application.

- **High-Speed, Parallelized Data Pipeline**  
  Engineered a robust web scraper using Python's `concurrent.futures` to process thousands of forum pages in parallel. This **increased data collection efficiency by over 10x** compared to sequential methods, reducing a multi-hour task to under an hour.

- **Resilient Scraping**  
  The scraper is fault-tolerant, with a built-in **retry mechanism and exponential backoff** to handle temporary server errors (`503` errors), ensuring a near-100% success rate on large-scale data collection.

- **Advanced RAG Pipeline**  
  Uses a sophisticated **"Retrieve & Re-rank"** strategy. After an initial vector search, a `CrossEncoder` model re-ranks the results for the highest contextual relevance, significantly improving the quality of the information sent to the LLM.

- **Conversational Memory**  
  The agent can handle natural, multi-turn conversations. It uses a **history-aware retriever** to intelligently reformulate follow-up questions into standalone queries, ensuring accurate retrieval without "prompt pollution."

- **Agentic Workflow & Guardrails**  
  - **Knowledge Gap Detector:** Self-critiques internal search results. If insufficient, it performs a **live web search** using the Tavily API.  
  - **Topic Guardrail:** Ensures the bot remains focused on its expertise, politely refusing to answer off-topic questions.

- **Resilient API Key Management**  
  Implemented a **"Use-Until-Exhausted" (Failover)** system to manage a pool of API keys. Automatically retries with backup keys if one is rate-limited.

- **Decoupled & Deployed Architecture**  
  The Streamlit app connects to a **hosted ChromaDB Cloud** vector store, decoupling the UI from data for **instant startup times** and a seamless user experience.

---

## Architecture Overview

The project is broken down into three main phases:

### 1. Phase 1: Data Pipeline (Scraping)

- `Anaplan Forums` → `Parallelized Scraper (Python)` → `Structured JSON Files`  
- Multi-threaded Python script scrapes specified tags, handling errors and saving structured Q&A data.

### 2. Phase 2: Knowledge Base Creation (Vectorization)

- `JSON Files` → `Document Chunker & Embedder (Python)` → `ChromaDB Cloud`  
- Processes raw data, splits long documents into smaller chunks, generates embeddings using `BAAI/bge-small-en-v1.5`, and uploads to ChromaDB.

### 3. Phase 3: RAG Application (UI & Logic)

- `User` → `Streamlit UI` → `LangChain Agent` → `Gemini LLM` → `Streamlit UI`  
- The Streamlit app captures user input, reformulates queries, retrieves & re-ranks context, and generates a sourced answer via Gemini 2.5 Flash-Lite.

---

## Tech Stack

- **Backend & Orchestration:** Python, LangChain  
- **AI & ML:** Google Gemini, Hugging Face Transformers (Sentence Transformers, Cross-Encoders)  
- **Database:** ChromaDB Cloud (Vector Store)  
- **Data Collection:** BeautifulSoup, Requests, Concurrent Futures  
- **Frontend & Deployment:** Streamlit, Streamlit Community Cloud  

---

## Local Setup & Installation

To run this project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Synapse-AI.git
cd Synapse-AI
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory and add your keys:

```env
# For the LLM
GOOGLE_API_KEYS="your_gemini_key_2,your_gemini_key_1,your_gemini_key_3..."

# For the hosted Vector DB
CHROMA_API_KEY="your_chroma_api_key"
CHROMA_TENANT="your_chroma_tenant_id"
CHROMA_DATABASE="your_chroma_database_name"

# For the Agentic Web Search
TAVILY_API_KEY="your_tavily_api_key"
```

### 5. Run the Application
```bash
streamlit run app.py
```

> The deployed version connects to a pre-built ChromaDB Cloud database. To build your own database locally, first run the scraping and vectorization scripts.

---

## Future Enhancements

- **Quantitative Evaluation Framework**  
  Add evaluation with `RAGAs` and a curated "golden dataset" to benchmark with metrics like faithfulness and relevancy.  

- **Automated Data Refresh**  
  Set up CI/CD with GitHub Actions to periodically scrape and update the knowledge base.  
