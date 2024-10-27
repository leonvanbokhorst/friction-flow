import numpy as np
from dataclasses import dataclass, field
import torch
import torch.autograd
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class NarrativeWave:
    """Represents a story as a quantum wave function"""

    content: str
    embedding: torch.Tensor  # Semantic embedding as quantum state
    amplitude: torch.Tensor  # Story strength/influence
    phase: torch.Tensor  # Story's phase in narrative space
    coherence: torch.Tensor  # Measure of story stability
    entanglement: Dict[str, float]  # Connections to other stories
    uncertainty: torch.Tensor = field(init=False)

    def __post_init__(self):
        assert self.embedding.dim() == 1, f"Embedding must be 1D, got shape {self.embedding.shape}"
        assert self.embedding.shape[0] == 768, f"Embedding must have 768 elements, got {self.embedding.shape[0]}"
        self.amplitude = self.amplitude.clone().detach().view(1)
        self.phase = self.phase.clone().detach().view(1)
        self.coherence = self.coherence.clone().detach().view(1)
        self.uncertainty = torch.tensor(np.random.random(), dtype=torch.float32).view(1)


class PatternMemory:
    def __init__(self):
        self.patterns = []
        self.pattern_strengths = torch.tensor([])
        
    def update_patterns(self, new_pattern, field_state):
        """Track and evolve observed patterns"""
        pattern_embedding = self.encode_pattern(new_pattern)
        
        if len(self.patterns) > 0:
            similarities = torch.cosine_similarity(
                pattern_embedding.unsqueeze(0),
                torch.stack(self.patterns)
            )
            
            if torch.any(similarities > 0.9):
                idx = torch.argmax(similarities)
                self.patterns[idx] = 0.9 * self.patterns[idx] + 0.1 * pattern_embedding
            else:
                self.patterns.append(pattern_embedding)
        else:
            self.patterns.append(pattern_embedding)
        
        # Update pattern strengths based on field state
        self.pattern_strengths = torch.tensor([
            torch.cosine_similarity(p.unsqueeze(0), field_state.unsqueeze(0))
            for p in self.patterns
        ])

    def encode_pattern(self, pattern: Dict) -> torch.Tensor:
        # Implement pattern encoding logic here
        # This is a placeholder implementation
        return torch.randn(768)  # Same dimension as story embeddings


class NarrativeFieldSimulator:
    def __init__(self):
        # Initialize quantum semantic space
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.quantum_dim = 768  # Embedding dimension
        self.stories: Dict[str, NarrativeWave] = {}
        self.field_state = torch.zeros(self.quantum_dim)
        self.total_energy = 1.0  # System's total energy
        self.energy_threshold = 2.0  # Maximum allowed energy
        self.pattern_memory = PatternMemory()

    def create_wave_function(self, content: str) -> NarrativeWave:
        """Convert story to quantum wave function"""
        # Create semantic embedding
        tokens = self.tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embedding = self.encoder(**tokens).last_hidden_state.mean(dim=1).squeeze(0)

        # Initialize quantum properties
        return NarrativeWave(
            content=content,
            embedding=embedding,
            amplitude=torch.tensor([1.0], dtype=torch.float32),
            phase=torch.tensor([np.random.random() * 2 * np.pi], dtype=torch.float32),
            coherence=torch.tensor([1.0], dtype=torch.float32),
            entanglement={},
        )

    def quantum_interference(self, wave1: NarrativeWave, wave2: NarrativeWave) -> float:
        """Enhanced quantum interference with uncertainty"""
        position_uncertainty = torch.std(wave1.embedding)
        momentum_uncertainty = torch.std(wave2.embedding)
        uncertainty_factor = 1.0 / (position_uncertainty * momentum_uncertainty)
        
        semantic_distance = torch.norm(wave1.embedding - wave2.embedding)
        tunnel_probability = torch.exp(-semantic_distance)
        
        similarity = torch.cosine_similarity(wave1.embedding.unsqueeze(0), 
                                           wave2.embedding.unsqueeze(0))
        phase_factor = torch.cos(wave1.phase - wave2.phase)
        
        return float(similarity * phase_factor * uncertainty_factor * tunnel_probability)

    def apply_field_effects(self, wave: NarrativeWave, dt: float):
        """Apply quantum field effects to a narrative wave"""
        # Simulate quantum evolution
        wave.phase += dt * wave.amplitude
        wave.coherence *= torch.exp(torch.tensor(-dt / 10.0))  # Gradual decoherence

        # Apply field interactions
        field_interaction = torch.cosine_similarity(
            wave.embedding.unsqueeze(0), self.field_state.unsqueeze(0)
        )
        wave.amplitude *= 1.0 + field_interaction * dt

        # Apply environmental effects
        env_embedding, vacuum_fluctuation = self.apply_environmental_effects(wave, dt)
        wave.embedding = 0.9 * wave.embedding + 0.1 * env_embedding
        self.field_state += vacuum_fluctuation

    def apply_environmental_effects(self, wave: NarrativeWave, dt: float):
        """Simulate interaction with environment"""
        noise = torch.randn_like(wave.embedding) * 0.01
        
        environment_coupling = torch.tensor(0.1, dtype=torch.float32)
        wave.coherence *= torch.exp(-environment_coupling * dt)
        
        vacuum_energy = 0.01
        vacuum_fluctuation = torch.randn_like(self.field_state) * vacuum_energy
        
        return wave.embedding + noise, vacuum_fluctuation

    def enforce_energy_conservation(self):
        """Enforce energy conservation in the field"""
        current_energy = torch.norm(self.field_state)
        if current_energy > self.energy_threshold:
            self.field_state = self.field_state * (self.total_energy / current_energy)
            scale_factor = torch.sqrt(self.total_energy / current_energy)
            for story in self.stories.values():
                story.amplitude *= scale_factor

    def update_field_state(self):
        """Update the overall field state based on all stories with non-linear effects"""
        contributions = torch.zeros(self.quantum_dim, dtype=torch.complex64)
        
        for story in self.stories.values():
            # Convert phase to tensor and combine wave functions with phase and amplitude
            phase_tensor = torch.tensor(story.phase, dtype=torch.float32)
            phase_factor = torch.complex(torch.cos(phase_tensor), torch.sin(phase_tensor))
            contribution = story.embedding * story.amplitude * phase_factor
            contributions += contribution
        
        new_field = contributions.requires_grad_()
        
        # Apply non-linear transformation
        field_potential = torch.tanh(new_field.real)
        
        # Calculate field gradient
        field_gradient = torch.autograd.grad(field_potential.sum(), new_field, create_graph=True)[0]
        
        # Combine linear and non-linear effects
        alpha = 0.7  # Adjustable parameter for balance between linear and non-linear effects
        self.field_state = alpha * new_field.real + (1 - alpha) * field_gradient.real
        
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

        # Consider pattern memory in emergence detection
        for i, pattern_embedding in enumerate(self.pattern_memory.patterns):
            similarity = torch.cosine_similarity(
                pattern_embedding.unsqueeze(0),
                embeddings
            )
            if torch.any(similarity > threshold):
                pattern = {
                    "stories": [story_keys[j] for j in (similarity > threshold).nonzero().squeeze(1)],
                    "coherence": float(similarity[similarity > threshold].mean()),
                    "field_strength": float(self.pattern_memory.pattern_strengths[i]),
                }
                patterns.append(pattern)

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

        # Enforce energy conservation
        self.enforce_energy_conservation()

        # Update pattern memory
        patterns = self.detect_emergence()
        for pattern in patterns:
            self.pattern_memory.update_patterns(pattern, self.field_state)


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
