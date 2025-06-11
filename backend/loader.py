# Data loader (PDFs, HTML, etc.)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import pandas as pd
import os
import glob
import logging
from app.rag.embedder import HuggingFaceEmbedder
from app.rag.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def lang_loader(self):
        logger.info(f"Processing file: {self.filepath}")
        if self.filepath.endswith(".pdf"):
            loader = PyPDFLoader(self.filepath)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} pages from PDF")
            return docs
        elif self.filepath.endswith(".txt"):
            loader = TextLoader(self.filepath)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents from text file")
            return docs
        else:
            logger.info(f"Skipping unsupported file type: {self.filepath}")
            return []
    
class Splitter:
        def __init__(self):
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
        def split(self, docs):
            return self.splitter.split_documents(docs)
    
class LoadLabels:
    def __init__(self, path):
        self.path = path
    
    def label_load_and_format(self):
        label_df = pd.read_csv(self.path)
        return label_df.set_index('filename').to_dict(orient = "index")

if __name__ == "__main__":
    # Example usage of the PDFLoader class
    # pdf_loader = PDFLoader(path="/Users/saksham/Desktop/RCC_RAG_Prototype/data/sample_docs/cancers-13-04751-v2.pdf")
    # documents = pdf_loader.load_and_split()
    # for doc in documents:
    #     print(doc.page_content)  # Print the content of each document chunk
    # print(f"Loaded {len(documents)} document chunks from {pdf_loader.path}")
    loader_obj = LoadLabels("/Users/saksham/Desktop/RCC_RAG_Prototype/data/sample_docs/sampleLabel.csv")
    label_dict = loader_obj.label_load_and_format()

    all_docs = []
    for filepath in glob.glob(os.path.join("/Users/saksham/Desktop/RCC_RAG_Prototype/data/sample_docs", "*")):
        docs = FileLoader(filepath).lang_loader()
        filename = os.path.basename(filepath)
        meta = label_dict.get(filename, {})
        meta["source"] = filename
        for doc in docs:
            doc.metadata.update(meta)
        all_docs.extend(docs)
    
    splitter = Splitter()
    split_docs = splitter.split(all_docs)
    
    embedder = HuggingFaceEmbedder().model
    vector_db = ChromaVectorStore(embedder)
    vector_db.add_documents(split_docs)

    results = vector_db.store.similarity_search("What is the name of this research paper?", k=3)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:200]}")

    vector_db.store.persist()
