import numpy as np
from dataclasses import dataclass, field
import torch
import torch.autograd
from typing import List, Dict, Any, Set
from transformers import AutoTokenizer, AutoModel
import logging
import torch.nn.functional as F
import torch.fft
import random
import pywt  # For wavelet analysis

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
            'pattern_coherence': torch.tensor([p['coherence'] for p in patterns] if patterns else [0.0]),
            'pattern_strength': torch.tensor([p['field_strength'] for p in patterns] if patterns else [0.0]),
            'pattern_radius': torch.tensor([p['radius'] for p in patterns] if patterns else [0.0])
        }
        self.trajectory.append(state_vector)
        
    def analyze_attractor(self) -> Dict:
        """Analyze attractor properties in phase space"""
        if len(self.trajectory) < 10:
            return {
                'field_stability': 0.0,
                'strength_stability': 0.0,
                'dominant_frequency': 0.0
            }
            
        recent_states = self.trajectory[-10:]
        
        field_variance = torch.var(torch.stack([s['field'] for s in recent_states]))
        
        # Handle case where pattern_strength might be empty
        strength_tensors = [s['pattern_strength'] for s in recent_states if s['pattern_strength'].numel() > 0]
        if strength_tensors:
            strength_variance = torch.var(torch.cat(strength_tensors))
            strength_series = torch.cat(strength_tensors)
            if strength_series.numel() > 1:
                fft = torch.fft.fft(strength_series)
                frequencies = torch.fft.fftfreq(len(strength_series))
                dominant_frequency = float(frequencies[torch.argmax(torch.abs(fft))])
            else:
                dominant_frequency = 0.0
        else:
            strength_variance = torch.tensor(0.0)
            dominant_frequency = 0.0
        
        # Avoid division by zero
        field_stability = float(1.0 / (1.0 + field_variance)) if field_variance != 0 else 1.0
        strength_stability = float(1.0 / (1.0 + strength_variance)) if strength_variance != 0 else 1.0
        
        return {
            'field_stability': field_stability,
            'strength_stability': strength_stability,
            'dominant_frequency': dominant_frequency
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


class FrequencyAnalyzer:
    """Enhance frequency analysis capabilities"""
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.frequency_history = []
        
    def analyze_frequencies(self, field_state: torch.Tensor, patterns: List[Dict]) -> Dict:
        """Multi-scale frequency analysis"""
        if not patterns:
            return {'dominant_frequencies': [], 'frequency_powers': []}
        
        signal = torch.tensor([p['field_strength'] for p in patterns])
        
        # Convert to numpy array for pywt
        np_signal = signal.numpy()
        
        # Determine appropriate decomposition level
        max_level = pywt.dwt_max_level(len(np_signal), 'db4')
        level = min(3, max_level)  # Use 3 or max_level, whichever is smaller
        
        if level == 0:
            # Not enough data for decomposition
            return {'dominant_frequencies': [1], 'frequency_powers': [float(np.mean(np_signal**2))]}
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(np_signal, 'db4', level=level)
        
        # Calculate power spectrum
        power_spectrum = [float(np.mean(np.abs(c)**2)) for c in coeffs]
        
        # Identify dominant frequencies
        dominant_scales = np.argsort(power_spectrum)[::-1]
        
        return {
            'dominant_frequencies': [2**(level-i) for i in dominant_scales],
            'frequency_powers': power_spectrum
        }


class StoryInteractionAnalyzer:
    """Enhanced story interaction analysis"""
    def __init__(self):
        self.interaction_history = []
        
    def analyze_interactions(self, stories: Dict[str, NarrativeWave]) -> Dict:
        """Analyze story interaction patterns"""
        story_ids = list(stories.keys())
        n_stories = len(story_ids)
        
        if n_stories < 2:
            return {}
            
        # Create interaction matrix
        interaction_matrix = torch.zeros((n_stories, n_stories))
        
        for i in range(n_stories):
            for j in range(i+1, n_stories):
                story1 = stories[story_ids[i]]
                story2 = stories[story_ids[j]]
                
                semantic_similarity = torch.cosine_similarity(
                    story1.embedding.unsqueeze(0),
                    story2.embedding.unsqueeze(0)
                )
                
                phase_coupling = torch.cos(story1.phase - story2.phase)
                amplitude_ratio = torch.min(story1.amplitude, story2.amplitude) / \
                                  torch.max(story1.amplitude, story2.amplitude)
                
                interaction_strength = (
                    semantic_similarity * 
                    phase_coupling * 
                    amplitude_ratio
                ).item()
                
                interaction_matrix[i, j] = interaction_matrix[j, i] = interaction_strength
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(interaction_matrix)
        except torch._C._LinAlgError:
            # Fallback to SVD if eigh fails
            U, S, V = torch.svd(interaction_matrix)
            eigenvalues, eigenvectors = S, V

        # Find interaction clusters
        clusters = []
        threshold = 0.7
        for i, vec in enumerate(eigenvectors.T):
            if eigenvalues[i] > threshold:
                cluster_members = [
                    story_ids[j] for j in range(n_stories) 
                    if abs(vec[j]) > 0.3
                ]
                if cluster_members:
                    clusters.append({
                        'strength': float(eigenvalues[i]),
                        'members': cluster_members
                    })
        
        return {
            'interaction_matrix': interaction_matrix.tolist(),
            'eigenvalues': eigenvalues.tolist(),
            'clusters': clusters
        }


class NarrativeFieldMetrics:
    """Track and analyze field metrics"""
    def __init__(self):
        self.metrics_history = []
        
    def update(self, field_state: torch.Tensor, stories: Dict[str, NarrativeWave], 
               patterns: List[Dict]) -> Dict:
        """Calculate comprehensive field metrics"""
        # Calculate field complexity (entropy)
        if field_state.dim() == 1:
            field_state = field_state.unsqueeze(0)  # Add a dimension if 1D
        
        normalized_field = torch.nn.functional.normalize(field_state.abs(), dim=-1)
        field_entropy = -torch.sum(normalized_field * torch.log(normalized_field + 1e-10))
        
        # Calculate story diversity
        if stories:
            embeddings = torch.stack([s.embedding for s in stories.values()])
            similarity_matrix = torch.cosine_similarity(
                embeddings.unsqueeze(1),
                embeddings.unsqueeze(0)
            )
            story_diversity = 1.0 - similarity_matrix.mean()
        else:
            story_diversity = 0.0
        
        # Calculate pattern stability
        if patterns:
            pattern_strengths = torch.tensor([p['field_strength'] for p in patterns])
            if len(pattern_strengths) > 1:
                pattern_stability = 1.0 / (1.0 + torch.std(pattern_strengths))
            else:
                pattern_stability = 1.0  # If there's only one pattern, consider it stable
        else:
            pattern_stability = 0.0
            
        metrics = {
            'field_entropy': float(field_entropy),
            'story_diversity': float(story_diversity),
            'pattern_stability': float(pattern_stability),
            'active_stories': len(stories),
            'active_patterns': len(patterns)
        }
        
        self.metrics_history.append(metrics)
        return metrics


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
        self.current_timestep = 0
        self.frequency_analyzer = FrequencyAnalyzer()
        self.interaction_analyzer = StoryInteractionAnalyzer()
        self.field_metrics = NarrativeFieldMetrics()

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
        """Enhanced field effects with better damping and stability"""
        # Phase evolution
        wave.phase += dt * wave.amplitude.clamp(max=10.0)  # Limit maximum phase change
        
        # Enhanced decoherence with amplitude-dependent damping
        decoherence_rate = torch.tensor(-dt / 10.0) * (1.0 + 0.1 * wave.amplitude.clamp(max=5.0))
        wave.coherence *= torch.exp(decoherence_rate).clamp(min=0.1, max=1.0)
        
        # Field interaction with damping
        field_interaction = torch.cosine_similarity(
            wave.embedding.unsqueeze(0), 
            self.field_state.unsqueeze(0)
        ).clamp(min=-1.0, max=1.0)
        
        # Damped amplitude evolution
        damping_factor = torch.exp(-0.01 * wave.amplitude.clamp(max=10.0))
        amplitude_change = (1.0 + field_interaction * dt) * damping_factor
        wave.amplitude *= amplitude_change.clamp(min=0.1, max=2.0)

        # Apply environmental effects
        env_embedding, vacuum_fluctuation = self.apply_environmental_effects(wave, dt)
        wave.embedding = 0.9 * wave.embedding + 0.1 * env_embedding
        self.field_state += vacuum_fluctuation

        # Ensure all tensor values are finite
        wave.amplitude = torch.where(torch.isfinite(wave.amplitude), wave.amplitude, torch.tensor([1.0]))
        wave.coherence = torch.where(torch.isfinite(wave.coherence), wave.coherence, torch.tensor([1.0]))
        wave.embedding = torch.where(torch.isfinite(wave.embedding), wave.embedding, torch.rand_like(wave.embedding) * 1e-6)

        # More gradual coherence change
        coherence_change = torch.randn(1) * 0.01  # Small random change
        wave.coherence += coherence_change
        wave.coherence = wave.coherence.clamp(min=0.1, max=1.0)

    def apply_environmental_effects(self, wave: NarrativeWave, dt: float):
        """Simulate interaction with environment using colored noise"""
        colored_noise = self.environmental_coupling.generate_colored_noise(wave.embedding.shape)
        
        environment_coupling = torch.tensor(0.1, dtype=torch.float32)
        wave.coherence *= torch.exp(-environment_coupling * torch.tensor(dt, dtype=torch.float32))
        
        vacuum_energy = torch.tensor(0.01, dtype=torch.float32)
        vacuum_fluctuation = colored_noise * vacuum_energy
        
        return wave.embedding + colored_noise, vacuum_fluctuation

    def enforce_energy_conservation(self):
        """More sophisticated energy conservation with nan handling"""
        if torch.isnan(self.field_state).any():
            logger.warning("NaN detected in field state. Resetting to small random values.")
            self.field_state = torch.rand_like(self.field_state) * 1e-6
        
        current_energy = torch.norm(self.field_state)
        if torch.isnan(current_energy) or current_energy == 0:
            logger.warning("Invalid current energy. Resetting field state.")
            self.field_state = torch.rand_like(self.field_state) * 1e-6
            current_energy = torch.norm(self.field_state)
        
        if current_energy > self.energy_threshold:
            # Calculate excess energy
            excess = current_energy - self.total_energy
            
            # Apply soft clamping using tanh
            damping = torch.tanh(excess / self.energy_threshold)
            scale_factor = (self.total_energy / current_energy) * (1.0 - damping)
            
            # Apply scaled correction
            self.field_state = self.field_state * scale_factor
            for story in self.stories.values():
                story.amplitude *= torch.sqrt(scale_factor)
        
        # Add small dissipation term
        dissipation = 0.01
        self.field_state *= (1.0 - dissipation)

    def update_field_state(self):
        """Update the overall field state based on all stories with non-linear effects and stability controls"""
        contributions = torch.zeros(self.quantum_dim, dtype=torch.complex64)
        
        for story in self.stories.values():
            phase_tensor = story.phase.clone().detach()  # Use clone().detach() instead of torch.tensor()
            phase_factor = torch.complex(torch.cos(phase_tensor), torch.sin(phase_tensor))
            contribution = story.embedding * story.amplitude * phase_factor
            contributions += contribution
        
        new_field = contributions.requires_grad_()
        
        # Apply non-linear transformation with stability control
        field_potential = torch.tanh(new_field.abs().clamp(min=-10, max=10))
        
        # Calculate field gradient with stability control
        field_gradient = torch.autograd.grad(field_potential.sum(), new_field, create_graph=True)[0]
        field_gradient_abs = field_gradient.abs().clamp(min=-1, max=1)
        field_gradient_angle = torch.angle(field_gradient)
        field_gradient = field_gradient_abs * torch.complex(torch.cos(field_gradient_angle), torch.sin(field_gradient_angle))
        
        # Combine linear and non-linear effects
        alpha = 0.7
        self.field_state = (alpha * new_field + (1 - alpha) * field_gradient).real
        
        # Ensure field state values are finite
        self.field_state = torch.where(torch.isfinite(self.field_state), self.field_state, torch.rand_like(self.field_state) * 1e-6)
        
        # Normalize field state
        norm = torch.norm(self.field_state)
        if norm > 0:
            self.field_state = self.field_state / norm
        else:
            self.field_state = torch.rand_like(self.field_state) * 1e-6

    def detect_emergence(self) -> List[Dict]:
        """Enhanced pattern detection with better uniqueness handling"""
        patterns = []
        processed_pairs: Set[Tuple[int, int]] = set()
        
        story_keys = list(self.stories.keys())
        
        # Check if there are any stories to process
        if not story_keys:
            logger.warning("No stories available for pattern detection")
            return patterns
        
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
        """Enhanced pattern interaction with stability controls"""
        distance = torch.norm(pattern1['center'] - pattern2['center'])
        overlap = torch.max(torch.zeros(1), 
                           (pattern1['radius'] + pattern2['radius'] - distance) / 
                           (pattern1['radius'] + pattern2['radius']))
        
        field_interaction = torch.cosine_similarity(
            pattern1['center'].unsqueeze(0),
            pattern2['center'].unsqueeze(0)
        ).clamp(min=-1, max=1)
        
        base_interaction = {
            'overlap': float(overlap),
            'field_interaction': float(field_interaction),
            'interaction_strength': float(overlap * field_interaction)
        }
        
        stability_factor = torch.exp(
            torch.tensor(-0.1 * (pattern1['field_strength'] + pattern2['field_strength']))
        ).clamp(min=0.1, max=1.0)
        
        base_interaction['interaction_strength'] *= float(stability_factor)
        base_interaction['stability'] = float(stability_factor)
        
        # Ensure all values are finite
        for key, value in base_interaction.items():
            if not np.isfinite(value):
                base_interaction[key] = 0.0
        
        return base_interaction

    def simulate_timestep(self, dt: float):
        """Simulate one timestep of field evolution with enhanced stability checks"""
        # Log the number of active stories at the beginning of each timestep
        logger.info(f"Number of active stories at start of timestep: {len(self.stories)}")

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
        initial_story_count = len(self.stories)
        self.stories = {k: v for k, v in self.stories.items() if v.coherence > 0.1}
        removed_stories = initial_story_count - len(self.stories)
        if removed_stories > 0:
            logger.info(f"Removed {removed_stories} fully decohered stories")

        # Dynamic story management
        if random.random() < 0.1:  # 10% chance each timestep
            if random.random() < 0.5 and len(self.stories) > 5:  # 50% chance to remove a story if more than 5 exist
                story_to_remove = random.choice(list(self.stories.keys()))
                del self.stories[story_to_remove]
                logger.info(f"Randomly removed story: {story_to_remove}")
            else:  # 50% chance to add a new story
                new_story = f"New research finding at timestep {self.current_timestep}"
                new_story_id = f"story_{len(self.stories)}"
                self.create_wave_function(new_story, new_story_id)
                logger.info(f"Added new story: {new_story}")

        # Enforce energy conservation
        self.enforce_energy_conservation()

        # Update pattern memory
        self.pattern_memory.story_dict = self.stories  # Update story_dict before detecting patterns
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
            field_energy = torch.norm(self.field_state)
            
            if torch.isnan(field_energy):
                logger.warning("NaN detected in field energy calculation.")
            
            logger.info(f"Attractor properties: {attractor_properties}")
            logger.info(f"Current field energy: {field_energy:.4f}")
            logger.info(f"Number of active stories: {len(self.stories)}")
            if self.stories:
                avg_coherence = sum(story.coherence.item() for story in self.stories.values()) / len(self.stories)
                logger.info(f"Average story coherence: {avg_coherence:.4f}")

        # Log the number of active stories at the end of the timestep
        logger.info(f"Number of active stories at end of timestep: {len(self.stories)}")

        # Global stability check
        for story in self.stories.values():
            story.amplitude = story.amplitude.clamp(min=0.1, max=10.0)
            story.coherence = story.coherence.clamp(min=0.1, max=1.0)
            story.embedding = torch.where(torch.isfinite(story.embedding), story.embedding, torch.rand_like(story.embedding) * 1e-6)
        
        self.field_state = torch.where(torch.isfinite(self.field_state), self.field_state, torch.rand_like(self.field_state) * 1e-6)
        self.field_state = self.field_state.clamp(min=-10, max=10)

        # Frequency analysis
        frequency_data = self.frequency_analyzer.analyze_frequencies(self.field_state, patterns)
        logger.info(f"Dominant frequencies: {frequency_data['dominant_frequencies']}")

        # Interaction analysis
        interaction_data = self.interaction_analyzer.analyze_interactions(self.stories)
        if interaction_data:
            logger.info(f"Detected {len(interaction_data['clusters'])} interaction clusters")

        # Comprehensive metrics
        metrics = self.field_metrics.update(self.field_state, self.stories, patterns)
        logger.info(f"Field metrics: {metrics}")


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
    simulator.current_timestep = t
    simulator.simulate_timestep(0.1)

    # Add a new story every 10 timesteps
    if t % 10 == 0:
        new_story = f"New research finding at timestep {t}"
        new_story_id = f"story_{len(simulator.stories)}"
        simulator.create_wave_function(new_story, new_story_id)
        logger.info(f"Added new story: {new_story}")

    # Check for emergent patterns
    if t % 10 == 0:
        patterns = simulator.detect_emergence()
        if patterns:
            logger.info(f"Timestep {t}: Detected {len(patterns)} emergent patterns")
            for i, pattern in enumerate(patterns):
                logger.info(f"  Pattern {i + 1}:")
                logger.info(f"    Stories: {', '.join(pattern['stories'])}")
                logger.info(f"    Coherence: {pattern['coherence']:.2f}")
                logger.info(f"    Field Strength: {pattern['field_strength']:.2f}")
        else:
            logger.info(f"Timestep {t}: No emergent patterns detected")

        # Analyze pattern effects on field
        field_energy = torch.norm(simulator.field_state)
        logger.info(f"Field energy: {field_energy:.2f}")

    logger.info(f"Number of active stories: {len(simulator.stories)}")















