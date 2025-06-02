from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class OpenAIEmbedder:
    def __init__(self):
        # Using a specific OpenAI model for embeddings
        # You can change the model name to any other supported by OpenAI
        # For example, "text-embedding-3-small" is a common choice
        # Ensure you have the OpenAI API key set in your environment
        # or in your configuration
        self.model = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed(self, texts):
        return self.model.embed_documents(texts)

class HuggingFaceEmbedder():
    def __init__(self):
        # Using a specific HuggingFace model for embeddings
        # You can change the model name to any other supported by HuggingFace
        # For example, "sentence-transformers/all-MiniLM-L6-v2" is a common choice
        self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts):
        return self.model.embed_documents(texts)

if __name__ == "__main__":
    # Example usage of the Embedder class
    embedder_instance = HuggingFaceEmbedder().model
    texts = ["Hello, world!", "This is a test document."]
    embeddings = embedder_instance.embed(texts)
    print("Embeddings:", embeddings)