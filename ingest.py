from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json
from langchain.schema import Document

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

# Function to load JSON files manually
def load_json_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert JSON data into LangChain Documents (modify as per JSON structure)
                if isinstance(data, list):  # Assuming JSON is a list of records
                    for record in data:
                        documents.append(Document(page_content=str(record)))
                else:  # Single JSON object
                    documents.append(Document(page_content=str(data)))
    return documents

# Function to create vector database
def create_vector_db():
    # Load PDF files
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    
   

    # Combine both PDF and JSON documents
    documents = pdf_documents 

    # Split texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={"device": "cpu"})

    # Create FAISS database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
