# FAISS/Chroma logic

from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class ChromaVectorStore:
    def __init__(self, embedder, path="./vector_db"):
        self.store = Chroma(
            collection_name="rag_docs",
            embedding_function=embedder,
            persist_directory=path
        )

    def add_texts(self, texts):
        self.store.add_texts(texts)

    def add_documents(self, documents):
        self.store.add_documents(documents)

    def similarity_search(self, query, k=5):
        return self.store.similarity_search(query, k=k)

class FAISSVectorStore:
    def __init__(self, embedder):
        self.store = FAISS.from_texts([], embedder)

    def add_texts(self, texts):
        self.store.add_texts(texts)

    def similarity_search(self, query, k=5):
        return self.store.similarity_search(query, k=k)
    
class VectorUtils:
    @staticmethod
    def delete_vector_db(path="./vector_db"):
        import shutil
        import os

        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted vector store at {path}")
        else:
            print(f"No vector store found at {path}. Nothing to delete.")
    
if __name__ == "__main__":
    VectorUtils.delete_vector_db(path="./vector_db")  # Clear existing vector store
    
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = ChromaVectorStore(embedder)
    
    # Load and add documents to the vector store
    from loader import PDFLoader
    pdf_loader = PDFLoader(path="/Users/saksham/Desktop/RCC_RAG_Prototype/data/sample_docs/Saksham_Pattem_Resume_SWE.pdf")
    documents = pdf_loader.load_and_split()
    vector_store.add_documents(documents)
    
    # Perform a similarity search
    query = "education"
    print("Query:", query)
    results = vector_store.similarity_search(query)
    print("Search results:", results[0].page_content)