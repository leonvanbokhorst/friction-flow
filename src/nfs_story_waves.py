import numpy as np
from dataclasses import dataclass
import torch
import torch.autograd
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class NarrativeWave:
    """Represents a story as a quantum wave function"""

    content: str
    embedding: torch.Tensor  # Semantic embedding as quantum state
    amplitude: float  # Story strength/influence
    phase: float  # Story's phase in narrative space
    coherence: float  # Measure of story stability
    entanglement: Dict[str, float]  # Connections to other stories

    def __post_init__(self):
        assert self.embedding.dim() == 1, f"Embedding must be 1D, got shape {self.embedding.shape}"
        assert self.embedding.shape[0] == 768, f"Embedding must have 768 elements, got {self.embedding.shape[0]}"


class NarrativeFieldSimulator:
    def __init__(self):
        # Initialize quantum semantic space
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.quantum_dim = 768  # Embedding dimension
        self.stories: Dict[str, NarrativeWave] = {}
        self.field_state = torch.zeros(self.quantum_dim)

    def create_wave_function(self, content: str) -> NarrativeWave:
        """Convert story to quantum wave function"""
        # Create semantic embedding
        tokens = self.tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embedding = self.encoder(**tokens).last_hidden_state.mean(dim=1).squeeze(0)

        # Initialize quantum properties
        return NarrativeWave(
            content=content,
            embedding=embedding,  # Now guaranteed to be 1D
            amplitude=1.0,
            phase=np.random.random() * 2 * np.pi,
            coherence=1.0,
            entanglement={},
        )

    def quantum_interference(self, wave1: NarrativeWave, wave2: NarrativeWave) -> float:
        """Calculate interference between two narrative waves"""
        # Compute quantum interference using cosine similarity and phase
        similarity = torch.cosine_similarity(wave1.embedding.unsqueeze(0), wave2.embedding.unsqueeze(0))
        phase_factor = np.cos(wave1.phase - wave2.phase)
        return float(similarity * phase_factor)

    def apply_field_effects(self, wave: NarrativeWave, dt: float):
        """Apply quantum field effects to a narrative wave"""
        # Simulate quantum evolution
        wave.phase += dt * wave.amplitude
        wave.coherence *= np.exp(-dt / 10.0)  # Gradual decoherence

        # Apply field interactions
        field_interaction = torch.cosine_similarity(
            wave.embedding, self.field_state.unsqueeze(0)
        )
        wave.amplitude *= 1.0 + field_interaction * dt

    def update_field_state(self):
        """Update the overall field state based on all stories with non-linear effects"""
        new_field = torch.zeros(self.quantum_dim, requires_grad=True)
        
        for story in self.stories.values():
            # Combine wave functions with phase and amplitude
            contribution = story.embedding * story.amplitude * torch.exp(1j * story.phase)
            new_field += contribution.real
        
        # Apply non-linear transformation
        field_potential = torch.tanh(new_field)
        
        # Calculate field gradient
        field_gradient = torch.autograd.grad(field_potential.sum(), new_field, create_graph=True)[0]
        
        # Combine linear and non-linear effects
        alpha = 0.7  # Adjustable parameter for balance between linear and non-linear effects
        self.field_state = alpha * new_field + (1 - alpha) * field_gradient
        
        # Normalize field state
        self.field_state = self.field_state / torch.norm(self.field_state)

    def detect_emergence(self) -> List[Dict]:
        """Detect emergent patterns in the narrative field"""
        logger.debug(f"Detecting emergence with {len(self.stories)} stories")
        patterns = []
        if len(self.stories) < 3:
            logger.debug("Not enough stories for meaningful patterns")
            return patterns

        story_keys = list(self.stories.keys())
        embeddings = torch.stack([self.stories[key].embedding for key in story_keys])
        
        similarity_matrix = torch.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0)
        )

        threshold = 0.8
        for i in range(len(similarity_matrix)):
            connected = (similarity_matrix[i] > threshold).nonzero().squeeze(1)
            if len(connected) > 2:
                pattern = {
                    "stories": [story_keys[j.item()] for j in connected if j.item() < len(story_keys)],
                    "coherence": float(similarity_matrix[i, connected].mean()),
                    "field_strength": float(
                        self.stories[story_keys[i]].amplitude
                    ),
                }
                patterns.append(pattern)
                logger.debug(f"Found pattern: {pattern}")

        logger.debug(f"Detected {len(patterns)} patterns")
        return patterns

    def simulate_timestep(self, dt: float):
        """Simulate one timestep of field evolution"""
        # Update all wave functions
        for story in self.stories.values():
            self.apply_field_effects(story, dt)

        # Calculate interference effects
        story_ids = list(self.stories.keys())
        for i in range(len(story_ids)):
            for j in range(i + 1, len(story_ids)):
                interference = self.quantum_interference(
                    self.stories[story_ids[i]], self.stories[story_ids[j]]
                )

                # Apply interference effects
                self.stories[story_ids[i]].amplitude *= 1.0 + interference * dt
                self.stories[story_ids[j]].amplitude *= 1.0 + interference * dt

                # Update entanglement
                self.stories[story_ids[i]].entanglement[story_ids[j]] = interference
                self.stories[story_ids[j]].entanglement[story_ids[i]] = interference

        # Update field state
        self.update_field_state()

        # Remove fully decohered stories
        self.stories = {k: v for k, v in self.stories.items() if v.coherence > 0.1}


# Example usage
simulator = NarrativeFieldSimulator()

# Add some initial stories
stories = [
    "The lab's funding was unexpectedly cut",
    "Dr. Patel's experiment showed promising results",
    "The department is considering a new research direction",
]

for i, content in enumerate(stories):
    wave = simulator.create_wave_function(content)
    simulator.stories[f"story_{i}"] = wave

# Run simulation
for t in range(100):
    simulator.simulate_timestep(0.1)

    # Check for emergent patterns
    if t % 10 == 0:
        patterns = simulator.detect_emergence()
        if patterns:
            print(f"Timestep {t}: Detected {len(patterns)} emergent patterns")
            for i, pattern in enumerate(patterns):
                print(f"  Pattern {i + 1}:")
                print(f"    Stories: {', '.join(pattern['stories'])}")
                print(f"    Coherence: {pattern['coherence']:.2f}")
                print(f"    Field Strength: {pattern['field_strength']:.2f}")
        else:
            print(f"Timestep {t}: No emergent patterns detected")

        # Analyze pattern effects on field
        field_energy = torch.norm(simulator.field_state)
        print(f"Field energy: {field_energy:.2f}")

    print(f"Number of active stories: {len(simulator.stories)}")
