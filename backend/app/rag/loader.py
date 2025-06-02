# Data loaders (PDFs, HTML, etc.)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFLoader:
    def __init__(self, path):
        self.path = path

    def load_and_split(self):
        loader = PyPDFLoader(self.path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return splitter.split_documents(docs)

if __name__ == "__main__":
    # Example usage of the PDFLoader class
    pdf_loader = PDFLoader(path="/Users/saksham/Desktop/RCC_RAG_Prototype/data/sample_docs/cancers-13-04751-v2.pdf")
    documents = pdf_loader.load_and_split()
    for doc in documents:
        print(doc.page_content)  # Print the content of each document chunk
    print(f"Loaded {len(documents)} document chunks from {pdf_loader.path}")