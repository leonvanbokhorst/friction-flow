import asyncio
from typing import Dict, Any

from ollama import Client as OllamaClient
from chromadb import Client as ChromaClient
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


class BrainLayer:
    """Manages multi-layer thought processing and memory operations.
    
    Handles initial perception, embedding generation, and memory comparison
    using local LLM and vector storage.
    """

    def __init__(self) -> None:
        # Initialize our local LLM
        self.llm = OllamaClient(host='http://localhost:11434')

        # Initialize our "memory"
        self.memory = ChromaClient()
        
        # Create collection with initial data
        self.collection = self.memory.create_collection(
            name="thought_memory",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Add some initial thoughts if collection is empty
        if len(self.collection.get()['ids']) == 0:
            initial_thoughts = [
                "Humans are conscious beings",
                "People feel emotions deeply",
                "We learn through experience"
            ]
            
            # Get embeddings for initial thoughts
            embeddings = []
            for thought in initial_thoughts:
                emb_response = self.llm.embeddings(
                    model="nomic-embed-text",
                    prompt=thought
                )
                embeddings.append(emb_response.embedding)
                
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=initial_thoughts,
                ids=[f"thought_{i}" for i in range(len(initial_thoughts))]
            )

    def process_thought(self, input_text: str) -> Dict[str, Any]:
        """Process input text through multiple cognitive layers.

        Args:
            input_text: The input text to process

        Returns:
            Dict containing current thought, related memories, and thought state
        """
        try:
            # Layer 1: Initial perception
            initial_response = self.llm.generate(
                model="llama3.2:latest",
                prompt=f"Process this thought: {input_text}"
            )

            # Layer 2: Get embeddings for memory comparison
            embeddings_response = self.llm.embeddings(
                model="nomic-embed-text",
                prompt=input_text
            )
            embeddings = embeddings_response.embedding

            # Layer 3: Compare with memory and enhance
            similar_thoughts = self.collection.query(
                query_embeddings=[embeddings],
                n_results=3,
                include=["documents", "distances"]  # Specify what to include in results
            )

            return {
                "current_thought": initial_response.response,
                "related_memories": {
                    "thoughts": similar_thoughts["documents"][0],
                    "distances": similar_thoughts["distances"][0]
                },
                "thought_state": embeddings[:10]  # Only show first 10 dimensions for brevity
            }
        except Exception as e:
            logger.error(f"Error processing thought: {str(e)}")
            raise


def main() -> None:
    """Main execution function."""
    brain = BrainLayer()
    result = brain.process_thought("I am a human")
    print(result)

if __name__ == "__main__":
    main()
