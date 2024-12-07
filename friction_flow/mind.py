class Mind:
    def __init__(self):
        # Add cache for thought embeddings
        self._thought_embeddings = {}
        # ... rest of init ...

    def _analyze_patterns(self) -> float:
        # Cache miss handling
        for thought in self.recent_thoughts:
            if thought.id not in self._thought_embeddings:
                self._thought_embeddings[thought.id] = self._get_embedding(thought.content)
            
        # Use cached embeddings
        embeddings = [self._thought_embeddings[t.id] for t in self.recent_thoughts]
        
        # Calculate similarity using cached embeddings
        if len(embeddings) >= 2:
            similarities = cosine_similarity(embeddings)
            return float(np.mean(similarities))
        return 0.0

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using the model.
        
        Args:
            text: The text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        # Implementation of embedding generation
        pass

    def add_thought(self, thought: Thought) -> None:
        """Add a new thought and cache its embedding."""
        self.recent_thoughts.append(thought)
        # Pre-cache the embedding
        self._thought_embeddings[thought.id] = self._get_embedding(thought.content) 