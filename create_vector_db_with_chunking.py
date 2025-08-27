import json
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import shutil

# --- Configuration ---
# The script will now only look for this single, combined JSON file.
JSON_FILE = 'all_anaplan_discussions.json'
DB_PATH = 'anaplan_db'
COLLECTION_NAME = 'anaplan_community'

# --- Chunking Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

print("🚀 Starting Phase 2: Creating Chunked Vector Database...")

# --- 1. Load Data from the Combined JSON file ---
try:
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        discussions = json.load(f)
    print(f"✅ Successfully loaded {len(discussions)} discussions from '{JSON_FILE}'.")
except FileNotFoundError:
    print(f"❌ The file '{JSON_FILE}' was not found. Please make sure your combined file is in this folder.")
    exit()
except json.JSONDecodeError:
    print(f"❌ The file '{JSON_FILE}' is empty or not a valid JSON file.")
    exit()

# --- 2. Prepare LangChain Documents ---
initial_docs = []
for item in discussions:
    question_body = item.get('question_body', '')
    if question_body:
        question_title = item.get('question_title', '')
        # Handle both accepted_answer (object) and accepted_answers (list) for flexibility
        accepted_answer_data = item.get('accepted_answer') or (item.get('accepted_answers') or [{}])[0]
        answer_text = accepted_answer_data.get('text', '')

        if answer_text:
            combined_text = f"Question: {question_title}\n\n{question_body}\n\nAccepted Answer: {answer_text}"
        else:
            combined_text = f"Article/Question: {question_title}\n\n{question_body}"

        metadata = {'url': item.get('question_url'), 'title': question_title}
        doc = Document(page_content=combined_text, metadata=metadata)
        initial_docs.append(doc)

# --- 3. Split Documents into Chunks ---
print(f"📄 Splitting {len(initial_docs)} documents into smaller chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunked_docs = text_splitter.split_documents(initial_docs)
print(f"✅ Created {len(chunked_docs)} chunks from the original documents.")

# --- 4. Create and Populate the Local Vector Database ---
print(f"\n✅ Preparing to add {len(chunked_docs)} chunks to the local vector database...")

# Delete the old database folder if it exists to ensure a clean build
if os.path.exists(DB_PATH):
    print(f"🗑️ Removing old database folder: '{DB_PATH}'")
    shutil.rmtree(DB_PATH)

embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Create the vector store from the chunked documents
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embedding_function,
    persist_directory=DB_PATH,
    collection_name=COLLECTION_NAME
)

print(f"✅ Successfully created and saved the chunked vector database locally at '{DB_PATH}'.")
print("\n🎉 Phase 2 complete! Your local database is now chunked and ready for upload.")
