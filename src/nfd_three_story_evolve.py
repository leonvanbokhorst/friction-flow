import sys
from pathlib import Path
import logging
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from language_models import LanguageModel, OllamaInterface
import asyncio
import time


@dataclass
class StoryState:
    """Captures the evolving state of a story over time"""

    resonance_level: float = 0.0
    active_themes: List[str] = field(default_factory=list)
    interaction_count: int = 0

    def update(self, resonance: float, new_themes: List[str]):
        self.resonance_level = 0.8 * self.resonance_level + 0.2 * resonance
        self.active_themes.extend(new_themes)
        self.interaction_count += 1


@dataclass
class Story:
    """Represents a single story in the narrative field"""

    id: str
    content: str
    embedding: np.ndarray  # Core semantic embedding
    memory_layer: List[Dict]  # Past interactions and experiences
    perspective_filter: np.ndarray  # What this story is attuned to notice
    position: np.ndarray  # Current position in field space
    velocity: np.ndarray  # Current movement vector
    themes: List[str]
    resonance_history: List[Dict]


class NarrativeFieldViz:
    """Handles visualization of field state"""

    def __init__(self, field_size: int = 1024):
        self.field_size = field_size
        self.history: List[Dict] = []
        self.logger = logging.getLogger(__name__)

    async def capture_state(self, field, timestep: int):
        """Capture current field state for visualization"""
        state = {
            "timestep": timestep,
            "story_positions": {
                story.id: story.position.copy() for story in field.stories
            },
            "story_velocities": {
                story.id: story.velocity.copy() for story in field.stories
            },
            "resonance_map": self._compute_resonance_map(field),
            "field_potential": field.field_potential.copy(),
        }
        self.history.append(state)
        self.logger.debug(f"Captured field state at timestep {timestep}")

    def _compute_resonance_map(self, field) -> Dict:
        """Compute resonance between all story pairs"""
        resonance_map = {}
        for i, story1 in enumerate(field.stories):
            for story2 in field.stories[i + 1 :]:
                resonance = field.detect_resonance(story1, story2)
                resonance_map[f"{story1.id}-{story2.id}"] = resonance
        return resonance_map


class NarrativeField:
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.stories: List[Story] = []
        self.field_memory = []
        self.collective_state = np.zeros(dimension)
        self.time = 0.0

        # Adjusted thresholds
        self.resonance_threshold = 0.3  # Lower threshold
        self.interaction_range = 3.0  # Increased range
        self.field_potential = np.zeros(dimension)
        self.logger = logging.getLogger(__name__)

    def detect_resonance(self, story1: Story, story2: Story) -> float:
        """Calculate resonance with improved theme handling"""
        # Get base embedding similarity
        embedding_similarity = float(
            F.cosine_similarity(
                torch.tensor(story1.embedding), torch.tensor(story2.embedding), dim=0
            )
        )

        # Enhanced theme handling
        shared_themes = set(story1.themes) & set(story2.themes)
        theme_overlap = len(shared_themes) / max(
            len(set(story1.themes) | set(story2.themes)), 1
        )

        # Weight shared themes more heavily
        theme_bonus = 0.2 if shared_themes else 0.0

        # Consider perspective filter alignment
        filter_alignment = float(
            F.cosine_similarity(
                torch.tensor(story1.perspective_filter),
                torch.tensor(story2.perspective_filter),
                dim=0,
            )
        )

        # Scale by distance more gradually
        distance = np.linalg.norm(story1.position - story2.position)
        distance_factor = 1.0 / (1.0 + distance / self.interaction_range)

        # Combine factors with theme bonus
        return (
            0.4 * embedding_similarity
            + 0.3 * (theme_overlap + theme_bonus)
            + 0.3 * filter_alignment
        ) * distance_factor

    def add_story(self, story: Story):
        """Add a new story to the field"""
        self.stories.append(story)
        self._update_field_potential()
        self.logger.info(f"Added new story: {story.id}")

    def _update_field_potential(self):
        """Update the field's potential energy based on story positions"""
        self.field_potential = np.zeros(self.dimension)
        for story in self.stories:
            # Each story contributes to the field potential
            self.field_potential += story.embedding * np.exp(
                -np.linalg.norm(story.position) / 10.0
            )

    def detect_resonance(self, story1: Story, story2: Story) -> float:
        """Calculate resonance between two stories"""
        # Base resonance on embedding similarity
        embedding_similarity = F.cosine_similarity(
            torch.tensor(story1.embedding), torch.tensor(story2.embedding), dim=0
        )

        # Consider theme overlap
        theme_overlap = len(set(story1.themes) & set(story2.themes)) / len(
            set(story1.themes) | set(story2.themes)
        )

        # Consider perspective filter alignment
        filter_alignment = F.cosine_similarity(
            torch.tensor(story1.perspective_filter),
            torch.tensor(story2.perspective_filter),
            dim=0,
        )

        # Scale resonance by distance
        distance = np.linalg.norm(story1.position - story2.position)
        distance_factor = np.exp(-distance / self.interaction_range)

        return float(
            (0.4 * embedding_similarity + 0.3 * theme_overlap + 0.3 * filter_alignment)
            * distance_factor
        )


class StoryPhysics:
    """Handles physical behavior of stories in the field"""

    def __init__(
        self,
        damping: float = 0.95,  # Increased damping
        attraction_strength: float = 0.1,  # Reduced strength
        max_force: float = 1.0,  # Force limiting
        max_velocity: float = 0.5,
    ):  # Velocity limiting
        self.damping = damping
        self.attraction_strength = attraction_strength
        self.max_force = max_force
        self.max_velocity = max_velocity
        self.logger = logging.getLogger(__name__)

    def _normalize_force(self, force: np.ndarray) -> np.ndarray:
        """Normalize force vector to prevent exponential growth"""
        magnitude = np.linalg.norm(force)
        if magnitude > self.max_force:
            return (force / magnitude) * self.max_force
        return force

    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Limit velocity magnitude"""
        magnitude = np.linalg.norm(velocity)
        if magnitude > self.max_velocity:
            return (velocity / magnitude) * self.max_velocity
        return velocity

    def update_story_motion(self, story: Story, field: NarrativeField, timestep: int):
        """Update story position and velocity based on field forces"""
        # Compute net force from other stories
        net_force = np.zeros(3)
        for other in field.stories:
            if other.id != story.id:
                # Compute force based on resonance
                resonance = field.detect_resonance(story, other)
                direction = other.position - story.position
                distance = np.linalg.norm(direction) + 1e-6  # Prevent division by zero

                # Scale force by distance with a minimum threshold
                force = self.attraction_strength * resonance * direction / distance
                net_force += force

        # Normalize and limit forces
        net_force = self._normalize_force(net_force)

        # Update velocity with damping
        story.velocity = self._limit_velocity(
            (1 - self.damping) * story.velocity + net_force
        )

        # Update position
        story.position += story.velocity

        self.logger.debug(
            f"Story {story.id} - "
            f"Position: {story.position}, "
            f"Velocity: {np.linalg.norm(story.velocity):.3f}"
        )


class EnhancedCollectiveStoryEngine:
    """Enhanced version with more sophisticated pattern detection"""

    def __init__(self, field: NarrativeField):
        self.field = field
        self.collective_memories = []
        self.story_states: Dict[str, StoryState] = {}
        self.logger = logging.getLogger(__name__)

    def update_story_states(self):
        """Update state tracking for all stories"""
        for story in self.field.stories:
            if story.id not in self.story_states:
                self.story_states[story.id] = StoryState()

            # Update based on recent interactions
            recent_memories = story.memory_layer[-5:]  # Look at last 5 interactions
            if recent_memories:
                avg_resonance = np.mean([m["resonance"] for m in recent_memories])
                recent_themes = [
                    theme for m in recent_memories for theme in m["themes"]
                ]
                self.story_states[story.id].update(avg_resonance, recent_themes)

    def detect_emergent_themes(self) -> List[str]:
        """Detect themes that are becoming more prominent"""
        theme_weights = {}

        for story_id, state in self.story_states.items():
            # Weight themes by story's resonance level
            for theme in state.active_themes:
                weight = state.resonance_level * (1 + state.interaction_count / 10)
                theme_weights[theme] = theme_weights.get(theme, 0) + weight

        # Return top themes sorted by weight
        sorted_themes = sorted(theme_weights.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, _ in sorted_themes[:3]]

    def generate_field_pulse(self, theme: str, intensity: float):
        """Generate a collective field pulse around a theme"""
        self.logger.info(
            f"Generating field pulse with theme '{theme}' and intensity {intensity}"
        )

        # Create pulse effect
        pulse = {
            "time": self.field.time,
            "theme": theme,
            "intensity": intensity,
            "affected_stories": [],
        }

        # Apply pulse to all stories in field
        for story in self.field.stories:
            if theme in story.themes:
                # Temporarily enhance story's resonance sensitivity
                story.perspective_filter *= 1 + intensity
                pulse["affected_stories"].append(story.id)

        self.collective_memories.append(pulse)


class ThemeEvolutionEngine:
    """Handles theme evolution and perspective shifts"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.theme_resonance = {}  # Track theme relationships

    def process_interaction(self, story1: Story, story2: Story, resonance: float):
        """Process theme interactions and evolution"""
        # Find theme relationships
        direct_shared = set(story1.themes) & set(story2.themes)
        story1_themes = set(story1.themes)
        story2_themes = set(story2.themes)

        # Theme relationship analysis
        relationships = {
            ("hope", "journey"): 0.7,
            ("loneliness", "discovery"): 0.6,
            ("guidance", "nature"): 0.5,
            ("duty", "freedom"): 0.4,
            ("imagination", "guidance"): 0.6,
            ("subconscious", "discovery"): 0.5,
        }

        # Find indirect theme relationships
        indirect_resonance = 0.0
        for t1 in story1_themes:
            for t2 in story2_themes:
                if (t1, t2) in relationships:
                    indirect_resonance += relationships[(t1, t2)]
                elif (t2, t1) in relationships:
                    indirect_resonance += relationships[(t2, t1)]

        # Calculate thematic impact
        theme_impact = len(direct_shared) * 0.3 + indirect_resonance * 0.2

        return {
            "direct_shared": direct_shared,
            "indirect_resonance": indirect_resonance,
            "theme_impact": theme_impact,
            "related_themes": [
                (t1, t2)
                for t1 in story1_themes
                for t2 in story2_themes
                if (t1, t2) in relationships or (t2, t1) in relationships
            ],
        }


class StoryInteractionEngine:
    """Enhanced interaction engine with better memory formation"""

    def __init__(self, field: NarrativeField):
        self.field = field
        self.logger = logging.getLogger(__name__)
        self.interaction_count = 0
        self.memory_threshold = 0.2  # Lower threshold for memory formation
        self.theme_engine = ThemeEvolutionEngine()

    def process_interaction(self, story1: Story, story2: Story):
        """Process interaction with theme evolution"""
        resonance = self.field.detect_resonance(story1, story2)
        
        # Process theme relationships
        theme_analysis = self.theme_engine.process_interaction(
            story1, story2, resonance
        )
        
        if resonance > self.memory_threshold:
            self.interaction_count += 1
            
            # Enhanced memory formation
            memory1 = {
                "time": self.field.time,
                "interacted_with": story2.id,
                "resonance": resonance,
                "themes": list(story2.themes),
                "shared_themes": list(theme_analysis['direct_shared']),
                "related_themes": theme_analysis['related_themes'],
                "interaction_id": self.interaction_count,
                "emotional_impact": resonance * (
                    len(theme_analysis['direct_shared']) + 
                    theme_analysis['indirect_resonance']
                )
            }

            memory2 = {
                "time": self.field.time,
                "interacted_with": story1.id,
                "resonance": resonance,
                "themes": list(story1.themes),
                "shared_themes": list(theme_analysis['direct_shared']),
                "interaction_id": self.interaction_count,
                "emotional_impact": resonance * len(theme_analysis['direct_shared']),
            }

            # Update memory layers
            story1.memory_layer.append(memory1)
            story2.memory_layer.append(memory2)

            # Log rich interaction details
            self.logger.info(
                f"\nRich Interaction #{self.interaction_count}:\n"
                f"  Stories: {story1.id} <-> {story2.id}\n"
                f"  Direct Shared Themes: {list(theme_analysis['direct_shared'])}\n"
                f"  Theme Relationships Found: {theme_analysis['related_themes']}\n"
                f"  Theme Impact: {theme_analysis['theme_impact']:.2f}\n"
                f"  Emotional Impact: {memory1['emotional_impact']:.2f}"
            )
            
            # Update perspectives with theme relationships
            self._update_perspective_filter(
                story1, story2, 
                theme_analysis['direct_shared'],
                theme_analysis['indirect_resonance']
            )

    def _update_perspective_filter(
        self, story1: Story, story2: Story, 
        shared_themes: set, indirect_resonance: float
    ):
        """Update perspective with theme relationships"""
        base_decay = 0.9
        direct_influence = len(shared_themes) * 0.15
        indirect_influence = indirect_resonance * 0.1
        
        # Combined theme influence
        decay = base_decay - (direct_influence + indirect_influence)
        
        # Update perspective
        new_perspective = (
            decay * story1.perspective_filter +
            (1 - decay) * story2.perspective_filter
        )
        
        # Record the shift
        shift_magnitude = np.sum(np.abs(new_perspective - story1.perspective_filter))
        story1.perspective_filter = new_perspective
        
        self.logger.info(
            f"Perspective Shift - {story1.id}:\n"
            f"  Magnitude: {shift_magnitude:.4f}\n"
            f"  Direct Theme Influence: {direct_influence:.2f}\n"
            f"  Indirect Theme Influence: {indirect_influence:.2f}"
        )


class StoryPhysics:
    """Handles physical behavior of stories in the field"""

    def __init__(self):
        # Physics parameters
        self.damping = 0.95  # Slightly reduced damping
        self.attraction_strength = 0.2  # Stronger attraction
        self.repulsion_strength = 0.1  # Add repulsion to prevent collapse
        self.min_distance = 0.5  # Minimum distance between stories
        self.interaction_range = 2.0  # Range for story interactions
        self.random_force = 0.05  # Small random force for exploration

        # Movement limits
        self.max_force = 0.3
        self.max_velocity = 0.2
        self.target_zone_radius = 10.0  # Desired story movement range

        self.logger = logging.getLogger(__name__)

    def update_story_motion(self, story: Story, field: NarrativeField, timestep: int):
        """Update story position and velocity with balanced forces"""
        net_force = np.zeros(3)

        # Forces from other stories
        for other in field.stories:
            if other.id != story.id:
                direction = other.position - story.position
                distance = np.linalg.norm(direction) + 1e-6
                direction_normalized = direction / distance

                # Resonance-based attraction
                resonance = field.detect_resonance(story, other)
                attraction = self.attraction_strength * resonance * direction_normalized

                # Distance-based repulsion
                repulsion = (
                    -self.repulsion_strength * direction_normalized / (distance**2)
                )
                if distance < self.min_distance:
                    repulsion *= 2.0  # Stronger repulsion when too close

                net_force += attraction + repulsion

        # Containment force - quadratic increase with distance
        displacement = story.position
        distance_from_center = np.linalg.norm(displacement)
        if distance_from_center > self.target_zone_radius:
            containment = (
                -0.1
                * (distance_from_center / self.target_zone_radius) ** 2
                * displacement
            )
            net_force += containment

        # Random exploration force - varies with time
        random_direction = np.random.randn(3)
        random_direction /= np.linalg.norm(random_direction)
        exploration_force = (
            self.random_force * random_direction * np.sin(timestep / 100)
        )
        net_force += exploration_force

        # Balance z-axis movement
        net_force[2] *= 0.3  # Reduce but don't eliminate z-axis movement

        # Apply force limits
        net_force = self._normalize_force(net_force)

        # Update velocity with damping
        story.velocity = self._limit_velocity(
            (1 - self.damping) * story.velocity + net_force
        )

        # Update position
        story.position += story.velocity

        # Log significant movements
        if timestep % 100 == 0:
            self.logger.debug(
                f"Story {story.id} at t={timestep}:\n"
                f"  Position: {story.position}\n"
                f"  Velocity: {np.linalg.norm(story.velocity):.3f}\n"
                f"  Force: {np.linalg.norm(net_force):.3f}"
            )

    def _normalize_force(self, force: np.ndarray) -> np.ndarray:
        """Normalize force vector to prevent exponential growth"""
        magnitude = np.linalg.norm(force)
        if magnitude > self.max_force:
            return (force / magnitude) * self.max_force
        return force

    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Limit velocity magnitude"""
        magnitude = np.linalg.norm(velocity)
        if magnitude > self.max_velocity:
            return (velocity / magnitude) * self.max_velocity
        return velocity

    def apply_field_constraints(self, stories: List[Story]):
        """Apply global constraints to all stories"""
        # Find center of mass
        com = np.mean([s.position for s in stories], axis=0)

        # If stories are drifting too far as a group, pull them back
        if np.linalg.norm(com) > 5.0:
            for story in stories:
                # Apply centering force proportional to distance from origin
                centering = -0.1 * story.position
                story.velocity += self._normalize_force(centering)
                story.velocity = self._limit_velocity(story.velocity)

    def _normalize_force(self, force: np.ndarray) -> np.ndarray:
        """Normalize force vector to prevent exponential growth"""
        magnitude = np.linalg.norm(force)
        if magnitude > self.max_force:
            return (force / magnitude) * self.max_force
        return force

    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Limit velocity magnitude"""
        magnitude = np.linalg.norm(velocity)
        if magnitude > self.max_velocity:
            return (velocity / magnitude) * self.max_velocity
        return velocity

    def apply_field_constraints(self, stories: List[Story]):
        """Apply global constraints to all stories"""
        # Find center of mass
        com = np.mean([s.position for s in stories], axis=0)

        # If stories are drifting too far as a group, pull them back
        if np.linalg.norm(com) > 5.0:
            for story in stories:
                # Apply centering force proportional to distance from origin
                centering = -0.1 * story.position
                story.velocity += self._normalize_force(centering)
                story.velocity = self._limit_velocity(story.velocity)


async def create_lighthouse_story(llm: LanguageModel) -> Story:
    """Create the lighthouse story with its properties"""
    content = """
    The Lighthouse on the Cliff
    This is the story of a lighthouse keeper who lights the beacon every night, 
    waiting for a ship that never comes. It contains themes of loneliness, duty, 
    and anticipation, with a constant longing for connection across the vast ocean.
    """
    embedding = await llm.generate_embedding(content)

    return Story(
        id="lighthouse",
        content=content,
        embedding=np.array(embedding),
        memory_layer=[],
        perspective_filter=np.ones(len(embedding)),
        position=np.random.randn(3) * 2.0,  # Scaled initial positions
        velocity=np.zeros(3),
        themes=["loneliness", "duty", "hope", "guidance"],
        resonance_history=[],
    )


async def create_path_story(llm: LanguageModel) -> Story:
    content = """
    The Path Through the Forest
    This story is about a traveler lost in a forest, searching for a way home. 
    It holds elements of uncertainty, hope, and resilience as the traveler 
    navigates unfamiliar terrain, feeling both wonder and isolation.
    """
    embedding = await llm.generate_embedding(content)
    return Story(
        id="path",
        content=content,
        embedding=np.array(embedding),
        memory_layer=[],
        perspective_filter=np.ones(len(embedding)),
        position=np.random.randn(3) * 2.0,  # Scaled initial positions
        velocity=np.zeros(3),
        themes=["journey", "discovery", "nature"],
        resonance_history=[],
    )


async def create_dream_story(llm: LanguageModel) -> Story:
    content = """
    The Child's Dream of Flight
    This story follows a child who dreams each night of flying, soaring above 
    villages, forests, and oceans. The dream is filled with freedom, innocence, 
    and limitless possibility, untouched by fear or doubt.
    """
    embedding = await llm.generate_embedding(content)
    return Story(
        id="dream",
        content=content,
        embedding=np.array(embedding),
        memory_layer=[],
        perspective_filter=np.ones(len(embedding)),
        position=np.random.randn(3) * 2.0,  # Scaled initial positions
        velocity=np.zeros(3),
        themes=["imagination", "freedom", "subconscious"],
        resonance_history=[],
    )


class StoryJourneyLogger:
    """Tracks and logs the journey of stories through the narrative field"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.journey_log = {}
        self.total_distances = {}  # Track cumulative distance for each story
        self.significant_events = []  # Track important moments

    def log_interaction(self, story1: Story, story2: Story, resonance: float):
        """Log a meaningful interaction between stories"""
        shared_themes = set(story1.themes) & set(story2.themes)
        distance = np.linalg.norm(story1.position - story2.position)

        # Log immediate interaction details
        self.significant_events.append(
            {
                "type": "interaction",
                "time": time.time(),
                "stories": (story1.id, story2.id),
                "resonance": resonance,
                "shared_themes": list(shared_themes),
                "distance": distance,
                "positions": {
                    story1.id: story1.position.copy(),
                    story2.id: story2.position.copy(),
                },
            }
        )

        self.logger.info(
            f"\nSignificant Interaction:\n"
            f"  {story1.id} <-> {story2.id}\n"
            f"  Resonance: {resonance:.2f}\n"
            f"  Shared Themes: {list(shared_themes)}\n"
            f"  Distance: {distance:.2f}\n"
            f"  Positions:\n"
            f"    {story1.id}: {story1.position}\n"
            f"    {story2.id}: {story2.position}\n"
        )

        # Track memory formation for both stories
        for story in [story1, story2]:
            if story.memory_layer:
                latest_memory = story.memory_layer[-1]
                self.logger.info(
                    f"\nMemory Formed - {story.id}:\n"
                    f"  Interaction with: {latest_memory['interacted_with']}\n"
                    f"  Resonance Level: {latest_memory['resonance']:.2f}\n"
                    f"  Themes Gained: {latest_memory['themes']}\n"
                    f"  Shared Themes: {latest_memory.get('shared_themes', [])}\n"
                )

    def log_story_state(self, story: Story, timestep: float):
        """Log detailed story state and track journey metrics"""
        if story.id not in self.journey_log:
            self.journey_log[story.id] = []
            self.total_distances[story.id] = 0.0

        # Calculate movement since last state
        if self.journey_log[story.id]:
            last_pos = self.journey_log[story.id][-1]["position"]
            movement = np.linalg.norm(story.position - last_pos)
            self.total_distances[story.id] += movement

            # Log significant movements
            if movement > 0.5:  # Threshold for significant movement
                self.significant_events.append(
                    {
                        "type": "movement",
                        "time": timestep,
                        "story_id": story.id,
                        "distance": movement,
                        "direction": story.velocity
                        / (np.linalg.norm(story.velocity) + 1e-6),
                    }
                )

        # Store current state
        state = {
            "timestep": timestep,
            "position": story.position.copy(),
            "velocity": story.velocity.copy(),
            "memory_count": len(story.memory_layer),
            "perspective_sum": story.perspective_filter.sum(),
            "total_distance": self.total_distances[story.id],
        }
        self.journey_log[story.id].append(state)

    def summarize_journey(self, story: Story):
        """Provide comprehensive journey summary"""
        journey = self.journey_log.get(story.id, [])
        if not journey:
            return

        start_state = journey[0]
        end_state = journey[-1]

        # Calculate journey metrics
        total_distance = self.total_distances[story.id]
        direct_distance = np.linalg.norm(
            end_state["position"] - start_state["position"]
        )
        wandering_ratio = total_distance / (direct_distance + 1e-6)

        # Memory and interaction analysis
        memories_formed = end_state["memory_count"]
        unique_interactions = len(set(m["interacted_with"] for m in story.memory_layer))

        # Perspective evolution
        perspective_change = (
            end_state["perspective_sum"] - start_state["perspective_sum"]
        )

        # Log comprehensive summary
        self.logger.info(
            f"\n=== Journey Summary for {story.id} ===\n"
            f"Movement Metrics:\n"
            f"  Total Distance Traveled: {total_distance:.2f}\n"
            f"  Direct Distance (start to end): {direct_distance:.2f}\n"
            f"  Wandering Ratio: {wandering_ratio:.2f}\n"
            f"\nInteraction Metrics:\n"
            f"  Memories Formed: {memories_formed}\n"
            f"  Unique Interactions: {unique_interactions}\n"
            f"  Perspective Shift: {perspective_change:.2f}\n"
            f"\nFinal State:\n"
            f"  Position: {end_state['position']}\n"
            f"  Velocity: {end_state['velocity']}\n"
            f"\nSignificant Events: {len([e for e in self.significant_events if e['type'] == 'interaction' and story.id in e['stories']])}\n"
        )


async def create_story_cluster():
    """Create initial story positions in a balanced configuration"""
    # Position stories in a triangle with some random offset
    base_positions = np.array(
        [[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]]
    )

    # Add random offset to make it interesting
    return base_positions + np.random.randn(3, 3) * 0.2


async def simulate_field():
    """Run a simulation of the narrative field"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting narrative field simulation")

    # Initialize language model
    llm = OllamaInterface()  # Using Ollama

    # Initialize simulation components
    field = NarrativeField()
    physics = StoryPhysics()
    visualizer = NarrativeFieldViz()
    interaction_engine = StoryInteractionEngine(field)
    collective_engine = EnhancedCollectiveStoryEngine(field)

    # Create stories
    lighthouse = await create_lighthouse_story(llm)
    logger.info(f"Created lighthouse story: {lighthouse.id}")
    path = await create_path_story(llm)
    logger.info(f"Created path story: {path.id}")
    dream = await create_dream_story(llm)
    logger.info(f"Created dream story: {dream.id}")

    # Add stories to field
    field.add_story(lighthouse)
    field.add_story(path)
    field.add_story(dream)

    # Add journey logger
    journey_logger = StoryJourneyLogger()

    # Simulation loop
    for t in range(1000):
        field.time = t

        # Update physics
        for story in field.stories:
            physics.update_story_motion(story, field, t)
            journey_logger.log_story_state(story, t)

        # Check for interactions
        for i, story1 in enumerate(field.stories):
            for story2 in field.stories[i + 1 :]:
                if np.linalg.norm(story1.position - story2.position) < 1.0:
                    interaction_engine.process_interaction(story1, story2)

        # Update story states
        collective_engine.update_story_states()

        # Capture visualization data
        await visualizer.capture_state(field, t)

        # Occasionally generate field pulses and detect patterns
        if t % 100 == 0:
            collective_engine.generate_field_pulse("seeking_connection", 0.5)
            emergent_themes = collective_engine.detect_emergent_themes()
            logger.info(f"Timestep {t} - Emergent themes: {emergent_themes}")

        # Update field state
        field._update_field_potential()

        # Log detailed state for each story
        for story in field.stories:
            journey_logger.log_story_state(story, t)

        # Check for and log interactions
        for i, story1 in enumerate(field.stories):
            for story2 in field.stories[i + 1 :]:
                distance = np.linalg.norm(story1.position - story2.position)
                if distance < field.interaction_range:
                    interaction_engine.process_interaction(story1, story2)
                    journey_logger.log_interaction(
                        story1, story2, field.detect_resonance(story1, story2)
                    )

    # Print final summaries
    logger.info("\n=== Final Journey Summaries ===")
    for story in field.stories:
        journey_logger.summarize_journey(story)

    logger.info("Narrative field simulation completed")

    # Clean up
    await llm.cleanup()


if __name__ == "__main__":
    asyncio.run(simulate_field())
