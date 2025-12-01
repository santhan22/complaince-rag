import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

def load_documents():
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            print(f"Loading: {path}")
            loader = PyPDFLoader(path)
            documents = loader.load()
            docs.extend(documents)
    return docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

def embed_and_store(chunks):
    print("Creating embeddings...")
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Storing to Chroma...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )
    db.persist()
    print("Done! Vector DB created.")

if __name__ == "__main__":
    print("Step 1: Loading documents...")
    docs = load_documents()

    print("Step 2: Splitting into chunks...")
    chunks = split_documents(docs)

    print(f"Created {len(chunks)} chunks.")

    print("Step 3: Embedding and saving...")
    embed_and_store(chunks)
