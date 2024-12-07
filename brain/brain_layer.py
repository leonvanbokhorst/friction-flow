import asyncio
from typing import Dict, Any, List, Union
from ollama import Client as OllamaClient
from chromadb import Client as ChromaClient
import numpy as np
from datetime import datetime
from logging import getLogger
import json

logger = getLogger(__name__)

class BrainLayer:
    """Enhanced brain layer with emotional processing and metacognition.
    
    Handles multiple layers of thought processing:
    - Initial perception
    - Embedding generation
    - Memory comparison
    - Emotional analysis
    - Metacognition
    - Temporal tracking
    """
    def __init__(self) -> None:
        # Initialize our local LLM and memory
        self.llm = OllamaClient(host='http://localhost:11434')
        self.memory = ChromaClient()
        
        # Track thought history for temporal analysis
        self.thought_history: List[Dict] = []
        
        # Create collection with initial data
        self.collection = self.memory.create_collection(
            name="thought_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize if empty
        self._initialize_memory()

    def _initialize_memory(self) -> None:
        """Initialize memory with base thoughts if empty."""
        if len(self.collection.get()['ids']) == 0:
            initial_thoughts = [
                "Humans are conscious beings",
                "People feel emotions deeply",
                "We learn through experience",
                "Time shapes our perceptions",
                "Memory influences decisions"
            ]
            
            embeddings = [
                self.llm.embeddings(
                    model="nomic-embed-text",
                    prompt=thought
                ).embedding
                for thought in initial_thoughts
            ]
            
            self.collection.add(
                embeddings=embeddings,
                documents=initial_thoughts,
                ids=[f"thought_{i}" for i in range(len(initial_thoughts))]
            )

    def add_emotional_layer(self, thought: str) -> Dict[str, Union[str, float]]:
        """Process the emotional content of a thought.
        
        Args:
            thought: Input thought to analyze
            
        Returns:
            Dictionary containing emotional analysis
        """
        try:
            emotional_prompt = (
                "Analyze the emotional components of this thought and respond "
                "with ONLY a JSON object containing two fields - 'primary_emotion' "
                f"and 'intensity' (0-1): {thought}"
            )
            
            response = self.llm.generate(
                model="llama3.2:latest",
                prompt=emotional_prompt
            )
            
            # Extract JSON from response
            emotion_data = json.loads(response.response)
            return emotion_data

        except Exception as e:
            logger.error(f"Error in emotional processing: {str(e)}")
            return {"primary_emotion": "neutral", "intensity": 0.0}

    def add_metacognition(self, 
                         thought: str, 
                         embeddings: List[float],
                         emotional_state: Dict) -> Dict:
        """Add self-reflective layer to thought processing.
        
        Args:
            thought: Original thought
            embeddings: Thought embeddings
            emotional_state: Result from emotional analysis
            
        Returns:
            Dictionary containing metacognitive analysis
        """
        # Calculate complexity metrics
        embedding_variance = np.var(embeddings)
        embedding_mean = np.mean(embeddings)
        
        # Analyze thought characteristics
        characteristics = {
            'thought_complexity': float(embedding_variance),
            'self_awareness_level': abs(float(embedding_mean)),
            'emotional_intensity': emotional_state['intensity'],
            'pattern_recognition': self._analyze_patterns(),
            'timestamp': datetime.now().isoformat()
        }
        
        return characteristics

    def _analyze_patterns(self) -> float:
        """Analyze patterns in recent thoughts."""
        if len(self.thought_history) < 2:
            return 0.0
            
        # Compare recent thoughts for patterns
        recent_embeddings = [t.get('thought_state', []) 
                           for t in self.thought_history[-3:]]
        if recent_embeddings:
            # Calculate similarity between recent thoughts
            similarities = np.corrcoef(recent_embeddings)
            return float(np.mean(similarities))
        return 0.0

    def process_thought(self, input_text: str) -> Dict[str, Any]:
        """Enhanced thought processing with all layers.
        
        Args:
            input_text: The input text to process
            
        Returns:
            Dictionary containing complete thought analysis
        """
        try:
            # Layer 1: Initial perception
            initial_response = self.llm.generate(
                model="llama3.2:latest",
                prompt=f"Process this thought: {input_text}"
            )

            # Layer 2: Generate embeddings
            embeddings_response = self.llm.embeddings(
                model="nomic-embed-text",
                prompt=input_text
            )
            embeddings = embeddings_response.embedding

            # Layer 3: Memory comparison
            similar_thoughts = self.collection.query(
                query_embeddings=[embeddings],
                n_results=3,
                include=["documents", "distances"]
            )

            # Layer 4: Emotional processing
            emotional_state = self.add_emotional_layer(input_text)

            # Layer 5: Metacognition
            metacognition = self.add_metacognition(
                input_text, 
                embeddings,
                emotional_state
            )

            # Compile complete thought state
            thought_state = {
                "current_thought": initial_response.response,
                "related_memories": {
                    "thoughts": similar_thoughts["documents"][0],
                    "distances": similar_thoughts["distances"][0]
                },
                "emotional_state": emotional_state,
                "metacognition": metacognition,
                "thought_state": embeddings[:10]  # First 10 dimensions
            }

            # Update thought history
            self.thought_history.append(thought_state)
            if len(self.thought_history) > 10:  # Keep last 10 thoughts
                self.thought_history.pop(0)

            return thought_state

        except Exception as e:
            logger.error(f"Error processing thought: {str(e)}")
            raise

def main() -> None:
    """Main execution function."""
    brain = BrainLayer()
    
    # Process a sequence of thoughts
    thoughts = [
        "I am a human",
        "I wonder about consciousness",
        "The universe is vast",
        "I want to sleep",
        "People are stupid",
    ]
    
    for thought in thoughts:
        print(f"\nProcessing thought: {thought}")
        print("-" * 50)
        result = brain.process_thought(thought)
        
        # Pretty print key components
        print(f"Emotional State: {result['emotional_state']}")
        print(f"Metacognition: {result['metacognition']}")
        print(f"Related Memories: {result['related_memories']['thoughts'][:2]}")
        print()

if __name__ == "__main__":
    main()