from typing import Optional, List, Dict, Any
from langchain.schema import BaseRetriever, Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever(BaseRetriever):
    vector_store: Any
    search_kwargs: Optional[Dict[str, Any]] = {"k": 4}
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        logger.info(f"Searching for query: {query}")
        docs = self.vector_store.similarity_search(
            query,
            filter=self.metadata,
            **(self.search_kwargs or {})
        )
        logger.info(f"Found {len(docs)} documents")
        for i, doc in enumerate(docs):
            logger.info(f"Document {i+1} content: {doc.page_content[:200]}...")
        return docs
