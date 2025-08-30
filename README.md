# Synapse AI: An Advanced RAG Chatbot for Community Forums

Synapse AI is a full-stack, **Retrieval-Augmented Generation (RAG) chatbot** designed to provide accurate, source-cited answers to user queries. It leverages a comprehensive knowledge base built by scraping thousands of real-world discussions from the **Anaplan Community forums**.  
This project demonstrates an **end-to-end AI/ML workflow**, from high-speed data collection and intelligent retrieval to a deployed, interactive web application.

---

## 🚀 Key Features

- **High-Speed Data Collection**  
  A parallelized Python scraper, optimized with multi-threading, gathers thousands of Q&A threads with an efficiency increase of over **10x** compared to sequential methods.

- **Intelligent Knowledge Base**  
  Uses a **ChromaDB vector store** and the `BAAI/bge-small-en-v1.5` embedding model to create a searchable database that understands the semantic meaning of text.

- **Advanced Re-ranking for High Accuracy**  
  Implements a **two-stage retrieval process**. After an initial broad search, a **Cross-Encoder model** re-ranks the results to find the most contextually relevant documents, significantly improving answer quality.

- **Accurate, Sourced Answers**  
  The **RAG pipeline** uses **LangChain** and **Google's Gemini 1.5 Flash LLM** to generate answers grounded in re-ranked forum data, effectively preventing AI hallucinations.

- **Scalable Architecture**  
  The application is **decoupled from the database**, which is hosted on **ChromaDB Cloud**, ensuring instant startup times for all users and a professional, scalable design.

---

## 🛠 Tech Stack

- **Languages & Frameworks**: Python, LangChain, Streamlit  
- **AI & ML**: Hugging Face Transformers (SentenceTransformers, Cross-Encoders), Gemini API  
- **Vector Database**: ChromaDB (Local and Cloud)  
- **Data Handling**: BeautifulSoup, Requests, JSON, Concurrent Futures  
- **Deployment & Tools**: Git, GitHub, Streamlit Community Cloud  

---

## 🏗 Project Architecture

The project is built in **three distinct phases**:

### 1. Data Scraping (`scraper.py`)
- Parallelized Python script efficiently extracts and structures thousands of Q&A threads.  
- Stores structured data into **JSON files**.

### 2. Knowledge Base Creation (`create_vector_db.py`)
- Processes raw JSON data.  
- Splits long documents into manageable chunks.  
- Converts them into embeddings using **BAAI/bge-small-en-v1.5**.  
- Uploads them to **ChromaDB Cloud**.

### 3. RAG Application & UI (`app.py`)
The Streamlit application executes the **RAG pipeline**:
1. **Retrieve** → Fast search in ChromaDB gathers broad candidate documents.  
2. **Re-rank** → A Cross-Encoder model refines the ranking for high relevance.  
3. **Generate** → The top-ranked documents are passed as context to **Gemini LLM**, which produces an **accurate, cited answer** in real time.

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Synapse-AI.git
cd Synapse-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your credentials
# Create a .env file in the root directory and add your API keys:
echo 'GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"' >> .env
echo 'CHROMA_API_KEY="YOUR_CHROMA_API_KEY"' >> .env
echo 'CHROMA_TENANT="YOUR_CHROMA_TENANT_ID"' >> .env
echo 'CHROMA_DATABASE="YOUR_CHROMA_DATABASE_NAME"' >> .env

# 4. Run the Streamlit app
streamlit run app.py
