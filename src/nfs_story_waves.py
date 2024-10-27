import numpy as np
from dataclasses import dataclass, field
import torch
import torch.autograd
from typing import List, Dict, Any, Set
from transformers import AutoTokenizer, AutoModel
import logging
import torch.nn.functional as F
import torch.fft

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
    def __init__(self, story_dict: Dict[str, NarrativeWave]):
        self.patterns: List[torch.Tensor] = []
        self.pattern_strengths: torch.Tensor = torch.tensor([])
        self.pattern_history: List[Dict[str, Any]] = []  # Track temporal evolution
        self.decay_rate: float = 0.95     # Pattern memory decay
        self.story_dict: Dict[str, NarrativeWave] = story_dict

    def update_patterns(self, new_pattern: Dict[str, Any], field_state: torch.Tensor) -> None:
        """Enhanced pattern tracking with temporal dynamics"""
        pattern_embedding = self.encode_pattern(new_pattern)
        pattern_time = len(self.pattern_history)
        
        # Apply temporal decay to existing patterns
        self.pattern_strengths *= self.decay_rate
        
        if len(self.patterns) > 0:
            similarities = torch.cosine_similarity(
                pattern_embedding.unsqueeze(0),
                torch.stack(self.patterns)
            )
            
            # Enhanced pattern merging with temporal weighting
            if torch.any(similarities > 0.9):
                idx = torch.argmax(similarities)
                age_factor = torch.exp(torch.tensor(-0.1 * (pattern_time - len(self.pattern_history))))
                self.patterns[idx] = (age_factor * self.patterns[idx] + 
                                    (1 - age_factor) * pattern_embedding)
                self.pattern_strengths[idx] += 1.0 - age_factor
            else:
                self.patterns.append(pattern_embedding)
                self.pattern_strengths = torch.cat([
                    self.pattern_strengths,
                    torch.tensor([1.0])
                ])
        else:
            self.patterns.append(pattern_embedding)
            self.pattern_strengths = torch.tensor([1.0])
        
        # Update temporal history
        self.pattern_history.append({
            'time': pattern_time,
            'embedding': pattern_embedding,
            'field_state': field_state.clone()
        })
        
        # Prune old patterns below strength threshold
        mask = self.pattern_strengths > 0.1
        self.patterns = [p for p, m in zip(self.patterns, mask) if m]
        self.pattern_strengths = self.pattern_strengths[mask]

    def encode_pattern(self, pattern: Dict[str, Any]) -> torch.Tensor:
        """Enhanced pattern encoding with semantic structure"""
        # Extract story embeddings
        story_embeddings = []
        for story_id in pattern['stories']:
            if story_id in self.story_dict:
                story_embeddings.append(self.story_dict[story_id].embedding)
            else:
                logger.warning(f"Story {story_id} not found in story_dict")
        
        if not story_embeddings:
            logger.warning("No valid story embeddings found for pattern")
            return torch.zeros(768)
            
        # Combine embeddings weighted by coherence
        combined = torch.stack(story_embeddings).mean(dim=0)
        
        # Add pattern metadata as modulation
        coherence_factor = torch.tensor(pattern['coherence'], dtype=torch.float32)
        strength_factor = torch.tensor(pattern['field_strength'], dtype=torch.float32)
        
        # Modulate combined embedding
        modulated = combined * coherence_factor * strength_factor
        
        return torch.nn.functional.normalize(modulated, dim=0)


class PatternEvolution:
    def __init__(self):
        self.pattern_trajectories: Dict[int, List[Dict]] = {}
        self.next_pattern_id = 0
    
    def update(self, current_patterns: List[Dict], field_state: torch.Tensor):
        new_trajectories = {}
        
        for pattern in current_patterns:
            best_match = None
            best_match_score = 0.8  # Minimum similarity threshold
            
            for pattern_id, trajectory in self.pattern_trajectories.items():
                if trajectory[-1]["active"]:  # Only consider active trajectories
                    similarity = self.calculate_pattern_similarity(
                        pattern, trajectory[-1]["pattern"]
                    )
                    if similarity > best_match_score:
                        best_match = pattern_id
                        best_match_score = similarity
            
            if best_match is not None:
                # Continue existing trajectory
                self.pattern_trajectories[best_match].append({
                    "pattern": pattern,
                    "time": len(self.pattern_trajectories[best_match]),
                    "field_state": field_state,
                    "active": True
                })
                new_trajectories[best_match] = True
            else:
                # Start new trajectory
                self.pattern_trajectories[self.next_pattern_id] = [{
                    "pattern": pattern,
                    "time": 0,
                    "field_state": field_state,
                    "active": True
                }]
                new_trajectories[self.next_pattern_id] = True
                self.next_pattern_id += 1
        
        # Mark unmatched trajectories as inactive
        for pattern_id in self.pattern_trajectories:
            if pattern_id not in new_trajectories:
                self.pattern_trajectories[pattern_id][-1]["active"] = False

    def calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        # Implement similarity calculation between patterns
        # This is a placeholder implementation
        return F.cosine_similarity(pattern1["center"].unsqueeze(0), 
                                   pattern2["center"].unsqueeze(0)).item()


class PhaseSpaceTracker:
    """Track system evolution in phase space"""
    def __init__(self, dim: int):
        self.trajectory = []
        self.phase_space_dim = dim
        
    def record_state(self, field_state: torch.Tensor, patterns: List[Dict]):
        """Record current state in phase space"""
        state_vector = {
            'field': field_state.clone(),
            'pattern_coherence': torch.tensor([p['coherence'] for p in patterns]),
            'pattern_strength': torch.tensor([p['field_strength'] for p in patterns]),
            'pattern_radius': torch.tensor([p['radius'] for p in patterns])
        }
        self.trajectory.append(state_vector)
        
    def analyze_attractor(self) -> Dict:
        """Analyze attractor properties in phase space"""
        if len(self.trajectory) < 10:
            return {}
            
        recent_states = self.trajectory[-10:]
        
        field_variance = torch.var(torch.stack([s['field'] for s in recent_states]))
        strength_variance = torch.var(torch.stack([s['pattern_strength'] for s in recent_states]))
        
        strength_series = torch.stack([s['pattern_strength'] for s in recent_states])
        fft = torch.fft.fft(strength_series)
        frequencies = torch.fft.fftfreq(len(strength_series))
        
        return {
            'field_stability': float(1.0 / (1.0 + field_variance)),
            'strength_stability': float(1.0 / (1.0 + strength_variance)),
            'dominant_frequency': float(frequencies[torch.argmax(torch.abs(fft))])
        }


class EnvironmentalCoupling:
    """Handle system-environment interactions"""
    def __init__(self, temperature: float = 0.1):
        self.temperature = torch.tensor(temperature, dtype=torch.float32)
        self.noise_history = []
        self.correlation_time = torch.tensor(10, dtype=torch.float32)
        
    def generate_colored_noise(self, shape: tuple) -> torch.Tensor:
        """Generate temporally correlated noise"""
        white_noise = torch.randn(shape)
        if not self.noise_history:
            self.noise_history = [white_noise]
            return white_noise
            
        alpha = torch.exp(-1.0 / self.correlation_time)
        colored_noise = alpha * self.noise_history[-1] + torch.sqrt(1 - alpha**2) * white_noise
        
        self.noise_history.append(colored_noise)
        if len(self.noise_history) > self.correlation_time.item():
            self.noise_history.pop(0)
            
        return colored_noise * self.temperature


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
        self.pattern_memory = PatternMemory(self.stories)
        self.pattern_evolution = PatternEvolution()
        self.phase_space_tracker = PhaseSpaceTracker(self.quantum_dim)
        self.environmental_coupling = EnvironmentalCoupling()

    def create_wave_function(self, content: str, story_id: str) -> NarrativeWave:
        """Convert story to quantum wave function and add to stories dict"""
        # Create semantic embedding
        tokens = self.tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embedding = self.encoder(**tokens).last_hidden_state.mean(dim=1).squeeze(0)

        # Initialize quantum properties
        wave = NarrativeWave(
            content=content,
            embedding=embedding,
            amplitude=torch.tensor([1.0], dtype=torch.float32),
            phase=torch.tensor([np.random.random() * 2 * np.pi], dtype=torch.float32),
            coherence=torch.tensor([1.0], dtype=torch.float32),
            entanglement={},
        )
        self.stories[story_id] = wave
        return wave

    def quantum_interference(self, wave1: NarrativeWave, wave2: NarrativeWave) -> float:
        """More sophisticated quantum interference calculation"""
        similarity = torch.cosine_similarity(wave1.embedding.unsqueeze(0), 
                                           wave2.embedding.unsqueeze(0))
        phase_factor = torch.cos(wave1.phase - wave2.phase)
        
        coherence_coupling = wave1.coherence * wave2.coherence
        
        position_uncertainty = torch.std(wave1.embedding)
        momentum_uncertainty = torch.std(wave2.embedding)
        uncertainty_factor = 1.0 / (position_uncertainty * momentum_uncertainty)
        
        semantic_distance = torch.norm(wave1.embedding - wave2.embedding)
        barrier_height = 1.0 / (wave1.amplitude * wave2.amplitude)
        tunnel_probability = torch.exp(-semantic_distance * barrier_height)
        
        entanglement_strength = torch.tensor(
            sum(set(wave1.entanglement.values()) & 
                set(wave2.entanglement.values()))
        )
        
        return float(
            similarity * 
            phase_factor * 
            coherence_coupling * 
            uncertainty_factor * 
            tunnel_probability * 
            (1 + entanglement_strength)
        )

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
        """Simulate interaction with environment using colored noise"""
        colored_noise = self.environmental_coupling.generate_colored_noise(wave.embedding.shape)
        
        environment_coupling = torch.tensor(0.1, dtype=torch.float32)
        wave.coherence *= torch.exp(-environment_coupling * torch.tensor(dt, dtype=torch.float32))
        
        vacuum_energy = torch.tensor(0.01, dtype=torch.float32)
        vacuum_fluctuation = colored_noise * vacuum_energy
        
        return wave.embedding + colored_noise, vacuum_fluctuation

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
        """Enhanced pattern detection with better uniqueness handling"""
        patterns = []
        processed_pairs: Set[Tuple[int, int]] = set()
        
        story_keys = list(self.stories.keys())
        embeddings = torch.stack([self.stories[key].embedding for key in story_keys])
        
        similarity_matrix = torch.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0)
        )

        threshold = 0.8
        
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if (i, j) in processed_pairs:
                    continue
                    
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    center = (embeddings[i] + embeddings[j]) / 2
                    radius = torch.norm(embeddings[i] - embeddings[j])
                    
                    distances = torch.norm(embeddings - center.unsqueeze(0), dim=1)
                    members = (distances <= radius * 1.2).nonzero().squeeze(1)
                    
                    if len(members) >= 2:
                        pattern = {
                            "stories": [story_keys[k.item()] for k in members],
                            "coherence": float(similarity),
                            "field_strength": float(
                                torch.mean(torch.stack([
                                    self.stories[story_keys[k.item()]].amplitude 
                                    for k in members
                                ]))
                            ),
                            "center": center,
                            "radius": radius
                        }
                        patterns.append(pattern)
                        
                        for k1 in members:
                            for k2 in members:
                                if k1 < k2:
                                    processed_pairs.add((k1.item(), k2.item()))

        return patterns

    def calculate_pattern_interaction(self, pattern1: Dict, pattern2: Dict) -> Dict:
        """Calculate interaction between patterns"""
        distance = torch.norm(pattern1['center'] - pattern2['center'])
        overlap = torch.max(torch.zeros(1), 
                           (pattern1['radius'] + pattern2['radius'] - distance) / 
                           (pattern1['radius'] + pattern2['radius']))
        
        phase_coherence = torch.cos(pattern1['phase'] - pattern2['phase'])
        
        field_interaction = torch.cosine_similarity(
            pattern1['center'].unsqueeze(0),
            pattern2['center'].unsqueeze(0)
        )
        
        return {
            'overlap': float(overlap),
            'phase_coherence': float(phase_coherence),
            'field_interaction': float(field_interaction),
            'interaction_strength': float(overlap * phase_coherence * field_interaction)
        }

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

        # Update pattern evolution
        self.pattern_evolution.update(patterns, self.field_state)

        # Record state in phase space
        self.phase_space_tracker.record_state(self.field_state, patterns)

        # Analyze pattern interactions
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], start=i+1):
                interaction = self.calculate_pattern_interaction(pattern1, pattern2)
                # Use interaction data to modify pattern evolution
                # This is a placeholder and should be implemented based on your specific requirements
                pattern1['field_strength'] *= (1 + interaction['interaction_strength'] * dt)
                pattern2['field_strength'] *= (1 + interaction['interaction_strength'] * dt)

        # Analyze attractor properties periodically
        if len(self.phase_space_tracker.trajectory) % 100 == 0:
            attractor_properties = self.phase_space_tracker.analyze_attractor()
            logger.info(f"Attractor properties: {attractor_properties}")


# Example usage
simulator = NarrativeFieldSimulator()

# Add some initial stories
stories = [
    "The lab's funding was unexpectedly cut",
    "Dr. Patel's experiment showed promising results",
    "The department is considering a new research direction",
]

for i, content in enumerate(stories):
    story_id = f"story_{i}"
    simulator.create_wave_function(content, story_id)

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

