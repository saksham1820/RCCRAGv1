from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from app.rag.loader import FileLoader
from app.rag.embedder import HuggingFaceEmbedder
from langchain.vectorstores import Chroma
from app.rag.generator import LlamaGenerator
from app.rag.retriever import Retriever
from app.core.config import VECTOR_DB_PATH
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    input: str

@router.post("/generate")
async def generate(request: QueryRequest):
    question = request.input
    logger.info(f"Received question: {question}")
    embedder = HuggingFaceEmbedder().model
    
    # Initialize vector store
    vector_db = Chroma(
        persist_directory="/Users/saksham/Desktop/RCC_RAG_Prototype/backend/app/rag/vector_db",
        embedding_function=embedder
    )    
    logger.info(f"Number of documents in vector DB: {vector_db._collection.count()}")


    # Setup retriever and generator
    retriever = Retriever(vector_store=vector_db, search_kwargs={"k": 4})
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

    try:
        final_response = retrieval_chain.invoke({"input": question})
        logger.info(f"Final response: {final_response}")
        return {"answer": final_response["answer"]}
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        return {"answer": f"Error occurred: {str(e)}"}
