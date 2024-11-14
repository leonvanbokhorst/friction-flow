from typing import List
import numpy as np

class VectorMemory:
    def __init__(self):
        self.memories = []
        self.embeddings = []
    
    async def similarity_search(self, query: str) -> List[str]:
        """Search for relevant memories based on query similarity."""
        if not self.memories:
            return []
            
        # TODO: Implement proper embedding and similarity search
        # For now, return all memories
        return self.memories
        
    async def add_memory(self, memory: str):
        """Add a new memory to the storage."""
        self.memories.append(memory)
        # TODO: Generate and store embedding 