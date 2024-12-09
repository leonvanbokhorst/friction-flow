import ollama
from chromadb import Client
import numpy as np


class BrainLayer:
    def __init__(self):
        # Initialize our local LLM
        self.llm = ollama.Client()

        # Initialize our "memory"
        self.memory = Client()
        self.collection = self.memory.create_collection("thought_memory")

    async def process_thought(self, input_text):
        # Layer 1: Initial perception
        initial_response = await self.llm.ask(
            "llama2", f"Process this thought: {input_text}"
        )

        # Layer 2: Get embeddings for memory comparison
        embeddings = await self.llm.embeddings("nomic-embed-text", input_text)

        # Layer 3: Compare with memory and enhance
        similar_thoughts = self.collection.query(
            query_embeddings=[embeddings], n_results=3
        )

        return {
            "current_thought": initial_response,
            "related_memories": similar_thoughts,
            "thought_state": embeddings,
        }

if __name__ == "__main__":
    brain = BrainLayer()
    print(brain.process_thought("I am a human"))
