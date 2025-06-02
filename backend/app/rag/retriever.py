# Similarity search + filtering
class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self):
        # results = self.vector_store.similarity_search(query, k=k)
        return self.vector_store.as_retriever()
