# LLM + prompt templates
import anthropic
import os


class ClaudeGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    def generate(self, query, docs):
        context = "\n\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=512,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class OpenAIGenerator:
    def __init__(self):
        from langchain.chat_models import ChatOpenAI
        self.model = ChatOpenAI(model="gpt-4", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, query, docs):
        context = "\n\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        return self.model.predict(prompt)

class LlamaGenerator:
    def __init__(self):
        from langchain_ollama import OllamaLLM
        self.model = OllamaLLM(model="llama3.1:8b", temperature=0.3)

    def generate(self, query, docs):
        context = "\n\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        return self.model.invoke(prompt)
    
if __name__ == "__main__":
    # Example usage of the generators
    query = "What is the capital of India?"
    docs = ["Paris is the capital of France.", "France is located in Europe."]

    # claude_gen = ClaudeGenerator()
    # print("Claude Response:", claude_gen.generate(query, docs))

    # openai_gen = OpenAIGenerator()
    # print("OpenAI Response:", openai_gen.generate(query, docs))

    llama_gen = LlamaGenerator()
    print("Llama Response:", llama_gen.generate(query, docs))