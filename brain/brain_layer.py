import asyncio
from typing import Dict, Any, List, Union, Optional, Tuple
from ollama import Client as OllamaClient
from chromadb import Client as ChromaClient
import numpy as np
from datetime import datetime
from logging import getLogger
import json
from enum import Enum
from functools import lru_cache

logger = getLogger(__name__)

# DON'T CHANGE THESE!!!
MODEL_NAME = "hermes3:latest" # don't change!!!
EMBEDDING_MODEL = "bge-m3:latest" # don't change!!!



class BrainState(Enum):
    """Enum for tracking brain states."""
    ACTIVE = "active"
    RESTING = "resting"
    LEARNING = "learning"
    REFLECTING = "reflecting"

class EmotionalMemory:
    """Tracks emotional patterns and transitions."""
    def __init__(self):
        self.emotional_history: List[Dict[str, Any]] = []
        self.max_history = 50

    def add_emotion(self, emotion: Dict[str, Any]) -> None:
        self.emotional_history.append({
            **emotion,
            'timestamp': datetime.now().isoformat()
        })
        if len(self.emotional_history) > self.max_history:
            self.emotional_history.pop(0)

    def get_emotional_trend(self) -> Dict[str, float]:
        """Analyze emotional patterns over time."""
        if not self.emotional_history:
            return {
                'stability': 0.0, 
                'average_intensity': 0.0,
                'emotional_momentum': 0.0
            }

        intensities = [e['intensity'] for e in self.emotional_history]
        valences = [e.get('valence', 0.0) for e in self.emotional_history]
        
        return {
            'stability': float(np.std(intensities)),
            'average_intensity': float(np.mean(intensities)),
            'emotional_momentum': float(np.mean(valences[-5:]) if len(valences) >= 5 else 0.0)
        }

class BrainLayer:
    """Enhanced brain layer with emotional processing, metacognition, and rest states."""
    
    def __init__(self) -> None:
        # Core components
        self.llm = OllamaClient(host='http://localhost:11434')
        self.memory = ChromaClient()
        self.emotional_memory = EmotionalMemory()
        
        # State tracking
        self.thought_history: List[Dict] = []
        self.brain_state = BrainState.ACTIVE
        self.rest_cycle_counter = 0
        self.max_thoughts_before_rest = 15
        
        # Create collection with initial data
        self.collection = self.memory.create_collection(
            name="thought_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        self._initialize_memory()

    def _initialize_memory(self) -> None:
        """Initialize memory with diverse base thoughts."""
        if len(self.collection.get()['ids']) == 0:
            initial_thoughts = [
                "Humans are conscious beings",
                "People feel emotions deeply",
                "We learn through experience",
                "Time shapes our perceptions",
                "Memory influences decisions",
                "Frustration leads to growth",
                "Joy comes from understanding",
                "Sleep refreshes the mind",
                "Anger clouds judgment",
                "Curiosity drives discovery",
                "Fear can protect or paralyze",
                "Love transforms perspective",
                "Doubt leads to questioning",
                "Hope sustains progress",
                "Wisdom comes from reflection"
            ]
            
            embeddings = [
                self.llm.embeddings(
                    model=EMBEDDING_MODEL,
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
        """Enhanced emotional processing with valence and context."""
        response_text = ""
        try:
            emotional_prompt = (
                "Analyze this thought and return ONLY a JSON object with format "
                "{'primary_emotion': string, 'intensity': float, 'valence': float, "
                "'context': string}. Intensity (0.0-1.0) based on strength. "
                f"Valence (-1.0 to 1.0) for positive/negative: {thought}"
            )
            
            response = self.llm.generate(
                model=MODEL_NAME,
                prompt=emotional_prompt
            )
            
            # Clean and validate response
            response_text = response.response.strip()
            try:
                emotion_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse emotional response: {response_text}")
                emotion_data = {
                    "primary_emotion": "neutral",
                    "intensity": 0.5,
                    "valence": 0.0,
                    "context": "parsing error"
                }
            
            self.emotional_memory.add_emotion(emotion_data)
            return emotion_data

        except Exception as e:
            logger.error(f"Error in emotional processing: {str(e)}")
            return {
                "primary_emotion": "neutral",
                "intensity": 0.0,
                "valence": 0.0,
                "context": "error in processing"
            }

    @lru_cache(maxsize=100)
    def _get_embedding(self, thought: str) -> List[float]:
        """Get embedding for a thought with LRU caching."""
        return self.llm.embeddings(
            model=EMBEDDING_MODEL,
            prompt=thought
        ).embedding

    def _analyze_thought_coherence(self, current_thought: str, prev_thoughts: List[str]) -> float:
        """Analyze thought coherence with history using cached embeddings."""
        if not prev_thoughts:
            return 1.0
        
        try:
            current_embedding = self._get_embedding(current_thought)
            prev_embeddings = [self._get_embedding(thought) for thought in prev_thoughts[-3:]]
            
            similarities = [
                np.dot(current_embedding, prev_emb) / 
                (np.linalg.norm(current_embedding) * np.linalg.norm(prev_emb))
                for prev_emb in prev_embeddings
            ]
            
            return float(np.mean(similarities))
        except Exception as e:
            logger.error(f"Error in coherence analysis: {str(e)}")
            return 0.0

    def _check_rest_state(self) -> Tuple[bool, str]:
        """Determine if brain needs to enter rest state."""
        self.rest_cycle_counter += 1
        
        if self.rest_cycle_counter >= self.max_thoughts_before_rest:
            self.rest_cycle_counter = 0
            self.brain_state = BrainState.RESTING
            return True, "Entering rest state for integration"
            
        emotional_trend = self.emotional_memory.get_emotional_trend()
        if emotional_trend['average_intensity'] > 0.8:
            return True, "High emotional intensity triggering rest"
            
        return False, ""

    def add_metacognition(self, 
                         thought: str, 
                         embeddings: List[float],
                         emotional_state: Dict) -> Dict:
        """Enhanced metacognitive processing."""
        # Calculate base metrics
        embedding_variance = np.var(embeddings)
        embedding_mean = np.mean(embeddings)
        
        # Get previous thoughts for coherence analysis
        prev_thoughts = [t['current_thought'] for t in self.thought_history[-3:]]
        coherence = self._analyze_thought_coherence(thought, prev_thoughts)
        
        # Get emotional trending
        emotional_trend = self.emotional_memory.get_emotional_trend()
        
        # Analyze thought characteristics
        characteristics = {
            'thought_complexity': float(embedding_variance),
            'self_awareness_level': abs(float(embedding_mean)),
            'emotional_intensity': emotional_state['intensity'],
            'emotional_valence': emotional_state.get('valence', 0.0),
            'pattern_recognition': self._analyze_patterns(),
            'coherence': coherence,
            'emotional_stability': emotional_trend['stability'],
            'emotional_momentum': emotional_trend['emotional_momentum'],
            'brain_state': self.brain_state.value,
            'timestamp': datetime.now().isoformat()
        }
        
        return characteristics

    def process_thought(self, input_text: str) -> Dict[str, Any]:
        """Enhanced thought processing with rest states and emotional tracking."""
        try:
            # Check rest state
            needs_rest, rest_reason = self._check_rest_state()
            if needs_rest:
                logger.info(f"Brain entering rest state: {rest_reason}")
                return {
                    "current_thought": "Processing and integrating recent experiences...",
                    "brain_state": self.brain_state.value,
                    "rest_reason": rest_reason
                }

            # Layer 1: Initial perception
            initial_response = self.llm.generate(
                model=MODEL_NAME,
                prompt=f"Process this thought: {input_text}"
            )

            # Layer 2: Generate embeddings
            embeddings_response = self.llm.embeddings(
                model=EMBEDDING_MODEL,
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
                "thought_state": embeddings[:10],
                "brain_state": self.brain_state.value
            }

            # Update thought history
            self.thought_history.append(thought_state)
            if len(self.thought_history) > 10:
                self.thought_history.pop(0)

            return thought_state

        except Exception as e:
            logger.error(f"Error processing thought: {str(e)}")
            raise

    def _analyze_patterns(self) -> float:
        """Analyze patterns usingcached embeddings."""
        if len(self.thought_history) < 2:
            return 0.0
        
        try:
            recent_thoughts = [t['current_thought'] for t in self.thought_history[-3:]]
            embeddings = [self._get_embedding(thought) for thought in recent_thoughts]
            
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
                    
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return 0.0
        
from typing import Dict, Any, List, Optional
import numpy as np

class StateAwareLLM:
    def __init__(self, brain_layer: BrainLayer):
        self.brain = brain_layer
        # Add rest cycle tracking
        self.rest_duration = 0
        self.max_rest_duration = 2  # Number of interactions before waking
        self.base_temperature = 0.7  # Add default base temperature
        
    def check_state_transition(self, brain_state: Dict) -> Optional[BrainState]:
        """Enhanced state transition logic with proper wake/sleep cycles."""
        current_state = self.brain.brain_state
        
        # Handle REST state special cases
        if current_state == BrainState.RESTING:
            self.rest_duration += 1
            
            # Check for wake triggers
            wake_words = ['wake', 'morning', 'hello', 'hey']
            if any(word in brain_state['current_thought'].lower() for word in wake_words) or \
               self.rest_duration >= self.max_rest_duration:
                self.rest_duration = 0
                return BrainState.ACTIVE
                
        # Check for sleep triggers
        sleep_words = ['sleep', 'tired', 'night', 'rest']
        if any(word in brain_state['current_thought'].lower() for word in sleep_words):
            return BrainState.RESTING
            
        return None
        
    def adjust_temperature(self, emotional_state: Dict) -> float:
        """Adjust temperature based on emotional state."""
        # Get values with defaults
        intensity = emotional_state.get('intensity', 0.5)
        valence = max(-1.0, min(1.0, emotional_state.get('valence', 0.0)))
        
        # More variable when emotional, more focused when neutral
        temperature = self.base_temperature + (intensity * 0.3)
        
        # Add slight randomness based on emotional valence
        temperature += valence * 0.1
        
        return min(max(temperature, 0.1), 1.0)  # Keep between 0.1 and 1.0

    def create_context_prompt(self, 
                            user_input: str, 
                            brain_state: Dict) -> str:
        """Create prompt with emotional and memory context."""
        
        # Get values with defaults if missing
        emotional_state = brain_state.get('emotional_state', {
            'primary_emotion': 'neutral',
            'intensity': 0.5
        })
        metacog = brain_state.get('metacognition', {
            'thought_complexity': 0.5,
            'emotional_intensity': 0.5
        })
        related_memories = brain_state.get('related_memories', {
            'thoughts': []
        })
        
        # Base contextual prompt
        context_parts = [
            f"Given your current emotional state of {emotional_state.get('primary_emotion', 'neutral')},",
            f"with an intensity of {emotional_state.get('intensity', 0.5)},",
        ]
        
        # Add memories if available
        if related_memories.get('thoughts'):
            memories = related_memories['thoughts'][:2]
            context_parts.append(f"and considering these related memories: {memories},")
        
        # Add the core prompt
        context_parts.extend([
            "respond to the following input naturally, letting your state influence your response style:",
            f"{user_input}"
        ])
        
        # Add state-specific modifications
        if metacog.get('thought_complexity', 0.5) > 0.7:
            context_parts.insert(0, "Taking a thoughtful, nuanced approach,")
        elif metacog.get('emotional_intensity', 0.5) > 0.8:
            context_parts.insert(0, "Responding with appropriate emotional depth,")
            
        return " ".join(context_parts)

    def generate_response(self, 
                         prompt: str, 
                         temperature: float) -> str:
        """Generate response using the LLM."""
        response = self.brain.llm.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": temperature}
        )
        return response.response

    def format_response(self, 
                       response: str, 
                       brain_state: Dict) -> str:
        """Format response based on current state."""
        # Get values with defaults
        emotional_state = brain_state.get('emotional_state', {
            'intensity': 0.5
        })
        metacog = brain_state.get('metacognition', {
            'emotional_intensity': 0.5
        })
        
        # Tired or resting state = shorter responses
        if metacog.get('emotional_intensity', 0.5) < 0.3 or brain_state.get('brain_state') == 'resting':
            response = ' '.join(response.split()[:50]) + "..."
            
        # High emotion = more expressive punctuation
        if emotional_state.get('intensity', 0.5) > 0.8:
            response = self.add_emotional_emphasis(response, emotional_state)
            
        return response

    def add_emotional_emphasis(self, 
                             response: str, 
                             emotional_state: Dict) -> str:
        """Add appropriate emotional emphasis to response."""
        valence = emotional_state.get('valence', 0.0)
        
        if valence > 0.8:  # Very positive
            response = response.replace('!', '!!').replace('.', '!')
        elif valence < -0.8:  # Very negative
            response = response.replace('!', '...').replace('.', '...')
        return response

    def respond(self, user_input: str) -> str:
        # Process through brain layer first
        brain_state = self.brain.process_thought(user_input)
        
        # Check for state transition
        new_state = self.check_state_transition(brain_state)
        if new_state:
            old_state = self.brain.brain_state
            self.brain.brain_state = new_state
            
            # Handle state transition messages
            if new_state == BrainState.ACTIVE and old_state == BrainState.RESTING:
                return "[Yawning and stretching...] Good morning! I'm awake and ready to chat! 😊"
            elif new_state == BrainState.RESTING:
                return "[Getting sleepy...] Mmm... time for a short rest. Wake me if you need me! 😴"
        
        # Add transition message
        transition_msg = f"\n[Current state: {self.brain.brain_state.value}]\n"
            
        # State-specific behaviors
        if self.brain.brain_state == BrainState.RESTING:
            return f"{transition_msg}[Taking a moment to process... breathing...]"
        elif self.brain.brain_state == BrainState.REFLECTING:
            prompt = f"Reflect deeply on: {user_input}"
        elif self.brain.brain_state == BrainState.LEARNING:
            prompt = f"Analyze patterns in: {user_input}"
        else:
            prompt = self.create_context_prompt(user_input, brain_state)
            
        # Generate and format response with error handling
        temperature = self.adjust_temperature(brain_state.get('emotional_state', {
            'intensity': 0.5,
            'valence': 0.0
        }))
        response = self.generate_response(prompt, temperature)
        final_response = self.format_response(response, brain_state)
        
        return f"{transition_msg}{final_response}"

def main():
    brain = BrainLayer()
    llm = StateAwareLLM(brain)
    
    print("Interactive Brain State Demo")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        print("\nProcessing...")
        print("-" * 50)
        
        response = llm.respond(user_input)
        print(f"Brain: {response}")
        
        # Show emotional trending
        trend = brain.emotional_memory.get_emotional_trend()
        print(f"\nEmotional Trend:")
        print(f"- Stability: {trend['stability']:.2f}")
        print(f"- Momentum: {trend['emotional_momentum']:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
