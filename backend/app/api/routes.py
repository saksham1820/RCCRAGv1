# from fastapi import APIRouter
# from pydantic import BaseModel
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from app.rag.loader import PDFLoader
# from app.rag.embedder import HuggingFaceEmbedder
# from langchain_community.vectorstores import Chroma
# from app.rag.generator import LlamaGenerator
# from app.rag.retriever import Retriever

# router = APIRouter()

# class QueryRequest(BaseModel):
#     input: str

# @router.post("/generate")
# async def generate(request: QueryRequest):
#     question = request.input

#     # Load document(s) - ideally done once and stored, but simplified here
#     loader = PDFLoader("/Users/saksham/Desktop/RCC_RAG_Prototype/data/sample_docs/Saksham_Pattem_Resume_SWE.pdf")
#     documents = loader.load_and_split()

#     # Create embeddings and vector store
#     embedder = HuggingFaceEmbedder().model
#     vectordb = Chroma.from_documents(documents, embedder)

#     # Setup retriever and generator
#     retriever_obj = Retriever(vectordb)
#     retriever = retriever_obj.retrieve()
#     generator = LlamaGenerator()

    
#     prompt = ChatPromptTemplate.from_template(
#     """You are a helpful assistant. Use the context below if it is relevant to answer the question. 
#     If the context is not helpful, feel free to answer based on your own knowledge.

#     <context>
#     {context}
#     </context>

#     Question: {input}"""
#     )

#     document_chain = create_stuff_documents_chain(
#         llm=generator.model,
#         prompt=prompt,
#     )

#     retrieval_chain = create_retrieval_chain(
#         retriever, document_chain
#     )

#     final_response = retrieval_chain.invoke({"input": question})
#     return {"answer": final_response["answer"]}

from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from app.rag.loader import PDFLoader
from app.rag.embedder import HuggingFaceEmbedder
from langchain_chroma import Chroma
from app.rag.generator import LlamaGenerator
from app.rag.retriever import Retriever
import os

router = APIRouter()

class QueryRequest(BaseModel):
    input: str

@router.post("/generate")
async def generate(request: QueryRequest):
    question = request.input
    persist_directory = "./vector_db"
    embedder = HuggingFaceEmbedder().model

    # Check if vector store already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("[INFO] Loading existing vector store...")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedder)
    else:
        print("[INFO] Creating and persisting new vector store...")
        loader = PDFLoader("/Users/saksham/Desktop/RCC_RAG_Prototype/data/sample_docs/Saksham_Pattem_Resume_SWE.pdf")
        documents = loader.load_and_split()
        vectordb = Chroma.from_documents(documents, embedder, persist_directory=persist_directory)
        vectordb.persist()

    # Setup retriever and generator
    retriever_obj = Retriever(vectordb)
    retriever = retriever_obj.retrieve()
    generator = LlamaGenerator()

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use the context below if it is relevant to answer the question. 
        If the context is not helpful, feel free to answer based on your own knowledge.

        <context>
        {context}
        </context>

        Question: {input}"""
    )

    document_chain = create_stuff_documents_chain(llm=generator.model, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    final_response = retrieval_chain.invoke({"input": question})
    return {"answer": final_response["answer"]}
