import sys
from pathlib import Path
import logging
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from language_models import LanguageModel, OllamaInterface
import asyncio
import time
import random
import json
import re

# Import the logging setup function
from logging_config import setup_logging

# Set up logging at the beginning of the script
setup_logging()

# Get a logger for this module
logger = logging.getLogger(__name__)


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


class ThemeRelationshipMap:
    """Manages theme relationships and their evolution"""

    def __init__(self):
        # Primary theme relationships with resonance values
        self.primary_relationships = {
            ("hope", "journey"): 0.7,
            ("loneliness", "discovery"): 0.6,
            ("guidance", "nature"): 0.5,
            ("duty", "freedom"): 0.4,
            ("imagination", "guidance"): 0.6,
            ("subconscious", "discovery"): 0.5,
            # Add more primary relationships
            ("hope", "freedom"): 0.6,
            ("loneliness", "nature"): 0.5,
            ("imagination", "discovery"): 0.8,
            ("journey", "nature"): 0.7,
        }

        # Secondary theme groups
        self.theme_groups = {
            "exploration": {"journey", "discovery", "nature"},
            "inner_world": {"imagination", "subconscious", "freedom"},
            "guidance": {"duty", "guidance", "hope"},
            "solitude": {"loneliness", "nature", "subconscious"},
        }
        self.logger = logging.getLogger(__name__)

    def get_theme_resonance(self, theme1: str, theme2: str) -> float:
        """Get resonance between two themes"""
        # Check direct relationship
        if (theme1, theme2) in self.primary_relationships:
            return self.primary_relationships[(theme1, theme2)]
        if (theme2, theme1) in self.primary_relationships:
            return self.primary_relationships[(theme2, theme1)]

        shared_groups = sum(
            1
            for group in self.theme_groups.values()
            if theme1 in group and theme2 in group
        )
        return 0.3 * shared_groups if shared_groups > 0 else 0.1


class StoryPerspective:
    """Manages a story's evolving perspective"""

    def __init__(self, initial_filter: np.ndarray):
        self.filter = initial_filter.copy()
        self.shift_history = []
        self.theme_influences = {}
        self.total_shift = 0.0
        self.logger = logging.getLogger(__name__)

    def update(
        self,
        other_filter: np.ndarray,
        shared_themes: set,
        indirect_resonance: float,
        theme_relationships: List[tuple],
    ):
        """Update perspective with detailed tracking"""
        # Calculate influence factors
        direct_influence = len(shared_themes) * 0.15
        indirect_influence = indirect_resonance * 0.1
        relationship_influence = len(theme_relationships) * 0.05

        total_influence = direct_influence + indirect_influence + relationship_influence
        decay = max(0.5, 0.9 - total_influence)  # Prevent complete override

        # Calculate new filter
        old_filter = self.filter.copy()
        self.filter = decay * self.filter + (1 - decay) * other_filter

        # Calculate shift magnitude
        shift = np.sum(np.abs(self.filter - old_filter))
        self.total_shift += shift

        # Record shift details
        self.shift_history.append(
            {
                "magnitude": shift,
                "direct_influence": direct_influence,
                "indirect_influence": indirect_influence,
                "relationship_influence": relationship_influence,
                "shared_themes": list(shared_themes),
                "theme_relationships": theme_relationships,
            }
        )

        # Update theme influences
        for theme in shared_themes:
            self.theme_influences[theme] = (
                self.theme_influences.get(theme, 0) + direct_influence
            )
        for t1, t2 in theme_relationships:
            self.theme_influences[t1] = (
                self.theme_influences.get(t1, 0) + relationship_influence
            )
            self.theme_influences[t2] = (
                self.theme_influences.get(t2, 0) + relationship_influence
            )

        return shift


class EmotionalState:
    def __init__(self, description: str, embedding: np.ndarray):
        self.description = description
        self.embedding = embedding

    async def update(
        self,
        other: Optional["EmotionalState"] = None,
        interaction_strength: float = 0.1,
        llm: Optional[LanguageModel] = None,
    ):
        if other is None or llm is None:
            # Simple update without interaction
            self.embedding += np.random.normal(
                0, interaction_strength, self.embedding.shape
            )
            return

        # Calculate the distance between emotional states
        distance = np.linalg.norm(self.embedding - other.embedding)

        # Generate a new emotional state description
        prompt = f"""
        Current emotional state: {self.description}
        Interacting emotional state: {other.description}
        Interaction strength: {interaction_strength}
        Distance between states: {distance}

        Based on this interaction, describe the new positive, neutral, or negative emotional state in 1 sentence:
        """
        new_description = await llm.generate(prompt)

        # Generate a new embedding for the updated emotional state
        new_embedding = await llm.generate_embedding(new_description)

        self.description = new_description
        self.embedding = np.array(new_embedding)

    def __str__(self):
        return self.description


class Story:
    def __init__(
        self,
        id: str,
        content: str,
        embedding: np.ndarray,
        perspective_filter: np.ndarray,
        themes: List[str],
        field: "NarrativeField",
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        emotional_state: EmotionalState = None,
        protagonist_name: str = None,
        **kwargs,
    ):
        self.id = id
        self.content = content
        self.embedding = embedding
        self.perspective_filter = perspective_filter
        self.themes = themes
        self.field = field
        self.position = position if position is not None else np.random.randn(3)
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.emotional_state = emotional_state
        self.previous_emotional_state = None
        self.memory_layer = []
        self.resonance_history = []
        self.total_perspective_shift = 0.0
        self.perspective_shifts = []
        self.protagonist_name = protagonist_name

        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return f"Story {self.id} ({self.protagonist_name}): {self.content[:50]}..."

    async def update_perspective(
        self,
        other_story: "Story",
        theme_impact: float,
        resonance: float,
        emotional_change: float,
        interaction_type: str,
    ) -> float:
        # Calculate perspective shift based on interaction factors
        shift = theme_impact * resonance * emotional_change * 0.1

        # Adjust shift based on interaction type
        if interaction_type == "conflict":
            shift *= -1  # Negative shift for conflicting interactions
        elif interaction_type == "collaboration":
            shift *= 1.5  # Larger positive shift for collaborative interactions

        # Update perspective filter
        self.perspective_filter += shift * np.random.randn(len(self.perspective_filter))

        return float(shift)  # Ensure we return a float

    def _calculate_emotional_influence(self, other: "Story") -> float:
        similarity = self.field._calculate_emotional_similarity(self, other)
        intensity = np.mean(
            [getattr(self.emotional_state, e) for e in vars(self.emotional_state)]
        )
        return similarity * intensity * 0.1

    async def update_emotional_state(
        self,
        other: "Story",
        interaction_type: str,
        resonance: float,
        llm: LanguageModel,
    ):
        self.previous_emotional_state = EmotionalState(
            self.emotional_state.description, self.emotional_state.embedding.copy()
        )
        await self.emotional_state.update(other.emotional_state, resonance, llm)

    def decay_emotions(self, decay_rate=0.01):
        for emotion in vars(self.emotional_state):
            current_value = getattr(self.emotional_state, emotion)
            decayed_value = max(0, current_value - decay_rate)
            setattr(self.emotional_state, emotion, decayed_value)

    def check_emotional_thresholds(self):
        if self.emotional_state.joy > 0.9:
            self.trigger_joyful_event()
        elif self.emotional_state.sadness > 0.8:
            self.trigger_melancholy_event()

    def evolve_themes(self, interaction_history):
        new_themes = set()
        for interaction in interaction_history[-10:]:  # Consider last 10 interactions
            if interaction["resonance"] > 0.5:
                new_themes.update(interaction["themes_gained"])
        self.themes = list(set(self.themes) | new_themes)[:5]  # Keep top 5 themes

    def calculate_interaction_strength(self, other_story):
        base_strength = self.field.detect_resonance(self, other_story)
        interaction_history = self.get_interaction_history(other_story)
        familiarity_bonus = min(0.2, len(interaction_history) * 0.02)
        return base_strength + familiarity_bonus

    async def respond_to_environmental_event(self, event: Dict[str, Any]) -> float:
        # Simple response to environmental events
        intensity = event.get("intensity", 0.5)

        # Update emotional state based on event intensity
        await self.emotional_state.update(interaction_strength=intensity * 0.2)

        # Add a memory of the event
        memory = {
            "type": "environmental_event",
            "event_name": event.get("name", "Unknown Event"),
            "intensity": intensity,
            "timestamp": self.field.time,
        }
        self.memory_layer.append(memory)

        return intensity

    async def update_state(self, avg_resonance: float, avg_shift: float) -> float:
        # Update the story's state based on recent interactions
        await self.emotional_state.update(interaction_strength=avg_resonance * 0.1)
        shift = avg_shift * 0.1
        self.perspective_filter += (
            shift  # Small perspective update based on average shift
        )
        return shift


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
            "emotional_states": {
                story.id: vars(story.emotional_state) for story in field.stories
            },
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
        self.resonance_threshold = 0.2  # Lower threshold to allow more interactions
        self.interaction_range = 3.0  # Increased range
        self.field_potential = np.zeros(dimension)
        self.logger = logging.getLogger(__name__)

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

        # Add emotional resonance
        emotional_similarity = self._calculate_emotional_similarity(story1, story2)

        # Scale resonance by distance
        distance = np.linalg.norm(story1.position - story2.position)
        distance_factor = np.exp(-distance / self.interaction_range)

        return float(
            (
                0.3 * embedding_similarity
                + 0.3 * theme_overlap
                + 0.2 * filter_alignment
                + 0.2 * emotional_similarity
            )
            * distance_factor
        )

    def _calculate_emotional_similarity(self, story1: Story, story2: Story) -> float:
        # Calculate cosine similarity between emotional state embeddings
        similarity = F.cosine_similarity(
            torch.tensor(story1.emotional_state.embedding),
            torch.tensor(story2.emotional_state.embedding),
            dim=0,
        )
        return float(similarity)

    def add_story(self, story: Story):
        """Add a new story to the field"""
        story.field = self  # Add this line
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

    async def apply_environmental_event(self, event: Dict[str, Any]):
        """Apply an environmental event to all stories in the field"""
        for story in self.stories:
            await story.respond_to_environmental_event(event)
        self.field_memory.append(event)


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
        self.collective_story = ""  # Add this line to initialize the collective story

    async def update_story_states(self):
        for story in self.field.stories:
            recent_memories = story.memory_layer[-5:]  # Get the 5 most recent memories
            if recent_memories:
                avg_resonance = np.mean(
                    [m.get("resonance", 0) for m in recent_memories]
                )
                perspective_shifts = []
                for m in recent_memories:
                    shift = m.get("perspective_shift", 0)
                    if asyncio.iscoroutine(shift):
                        shift = await shift
                    perspective_shifts.append(shift)

                avg_shift = np.mean(perspective_shifts)

                # Update story state based on recent interactions
                shift = await story.update_state(avg_resonance, avg_shift)
                story.total_perspective_shift += shift

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

    def write_collective_story(self):
        """Generate and update the collective story based on current field state"""
        emergent_themes = self.detect_emergent_themes()
        story_summaries = [self.summarize_story(story) for story in self.field.stories]

        collective_narrative = (
            f"The narrative field pulses with {len(self.field.stories)} stories. "
        )
        collective_narrative += f"Emergent themes of {', '.join(emergent_themes)} weave through the collective consciousness. "

        for summary in story_summaries:
            collective_narrative += f"{summary} "

        self.collective_story = collective_narrative
        self.logger.info(f"Updated collective story: {self.collective_story[:100]}...")

    def summarize_story(self, story: Story) -> str:
        """Generate a brief summary of a story's current state"""
        return f"Story {story.id} resonates with {', '.join(story.themes[:3])}, its journey marked by {len(story.memory_layer)} memories."


class ThemeEvolutionEngine:
    """Handles theme evolution and perspective shifts"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.theme_resonance = {}  # Track theme relationships
        self.logger = logging.getLogger(__name__)


class EnhancedInteractionEngine:
    # ... (other methods remain the same)

    async def process_interaction(self, story1: Story, story2: Story):
        if story1.id == story2.id:
            return  # Prevent a story from interacting with itself

        interaction_type = await self.determine_interaction_type(story1, story2)
        resonance = self.field.detect_resonance(story1, story2)

        if resonance > self.field.resonance_threshold:
            theme_impact = self.calculate_theme_impact(story1, story2)
            emotional_change = self._calculate_emotional_influence(story1, story2)

            perspective_shift = await story1.update_perspective(
                story2, theme_impact, resonance, emotional_change, interaction_type
            )

            await story1.update_emotional_state(
                story2, interaction_type, resonance, self.llm
            )

            # Add memory of interaction
            memory = {
                "type": "interaction",
                "partner_id": story2.id,
                "resonance": resonance,
                "interaction_type": interaction_type,
                "perspective_shift": perspective_shift,
                "timestamp": self.field.time,
            }
            story1.memory_layer.append(memory)

            return resonance, interaction_type
        return 0, None

    # ... (rest of the class remains the same)


class StoryInteractionEngine:
    def __init__(self, field: NarrativeField):
        self.field = field
        self.logger = logging.getLogger(__name__)

    async def process_interaction(self, story1: Story, story2: Story):
        # Basic interaction processing
        resonance = self.field.detect_resonance(story1, story2)
        if resonance > self.field.resonance_threshold:
            # Perform basic interaction logic here
            pass


class EnhancedInteractionEngine(StoryInteractionEngine):
    def __init__(self, field: NarrativeField, llm: LanguageModel):
        super().__init__(field)
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    async def determine_interaction_type(self, story1: Story, story2: Story) -> str:
        prompt = f"""
        Story 1 Themes: {', '.join(story1.themes)}
        Story 1 Emotional State: {story1.emotional_state}
        
        Story 2 Themes: {', '.join(story2.themes)}
        Story 2 Emotional State: {story2.emotional_state}
        
        Based on the themes and emotional states of these two stories, what type of interaction might occur between them?
        Choose from: collaboration, conflict, inspiration, reflection, transformation, challenge, or synthesis.
        
        Respond only with one chosen interaction type as a SINGLE WORD response.
        """
        return await self.llm.generate(prompt)

    async def process_interaction(self, story1: Story, story2: Story):
        if story1.id == story2.id:
            return  # Prevent a story from interacting with itself

        interaction_type = await self.determine_interaction_type(story1, story2)
        resonance = self.field.detect_resonance(story1, story2)

        if resonance > self.field.resonance_threshold:
            theme_impact = self.calculate_theme_impact(story1, story2)
            emotional_change = self._calculate_emotional_influence(story1, story2)

            perspective_shift = await story1.update_perspective(
                story2, theme_impact, resonance, emotional_change, interaction_type
            )

            await story1.update_emotional_state(
                story2, interaction_type, resonance, self.llm
            )

            # Add memory of interaction
            memory = {
                "type": "interaction",
                "partner_id": story2.id,
                "resonance": resonance,
                "interaction_type": interaction_type,
                "perspective_shift": perspective_shift,  # This is now awaited
                "timestamp": self.field.time,
            }
            story1.memory_layer.append(memory)

            return resonance, interaction_type
        return 0, None

    def calculate_theme_impact(self, story1: Story, story2: Story) -> float:
        shared_themes = set(story1.themes) & set(story2.themes)
        return len(shared_themes) / max(len(story1.themes), len(story2.themes))

    def _calculate_emotional_influence(self, story1: Story, story2: Story) -> float:
        # Implement the emotional influence calculation here
        # This is a placeholder implementation
        return 0.5  # Return a value between 0 and 1


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


class StoryJourneyLogger:
    """Tracks and logs the journey of stories through the narrative field"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.journey_log = {}
        self.total_distances = {}  # Track cumulative distance for each story
        self.significant_events = []  # Track important moments

    def log_interaction(
        self, story1: Story, story2: Story, resonance: float, interaction_type: str
    ):
        latest_memory = story1.memory_layer[-1] if story1.memory_layer else {}
        perspective_shift = latest_memory.get("perspective_shift", 0)

        # If perspective_shift is a coroutine, we need to run it in an event loop
        if asyncio.iscoroutine(perspective_shift):
            perspective_shift = asyncio.get_event_loop().run_until_complete(
                perspective_shift
            )

        log_entry = (
            f"Interaction between {story1.id} and {story2.id}:\n"
            f"  Resonance: {resonance:.2f}\n"
            f"  Interaction Type: {interaction_type}\n"
            f"  Perspective Shift: {perspective_shift:.4f}\n"
            f"  Shared Themes: {set(story1.themes) & set(story2.themes)}\n"
            f"  Distance: {np.linalg.norm(story1.position - story2.position):.2f}\n"
            f"  Positions:\n"
            f"    {story1.id}: {story1.position}\n"
            f"    {story2.id}: {story2.position}\n"
        )
        self.logger.info(log_entry)

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
        """Enhanced journey summary with accumulated perspective shifts"""
        journey = self.journey_log.get(story.id, [])
        if not journey:
            return

        start_state = journey[0]
        end_state = journey[-1]

        # Calculate metrics
        total_distance = self.total_distances[story.id]
        direct_distance = np.linalg.norm(
            end_state["position"] - start_state["position"]
        )
        wandering_ratio = total_distance / (direct_distance + 1e-6)

        # Perspective analysis
        significant_shifts = [
            s for s in story.perspective_shifts if s["magnitude"] > 0.01
        ]
        avg_shift = (
            np.mean([s["magnitude"] for s in significant_shifts])
            if significant_shifts
            else 0
        )

        # Safely get unique interactions
        unique_interactions = {
            m["interacted_with"]
            for m in story.memory_layer
            if "interacted_with" in m
        }
        num_unique_interactions = len(unique_interactions)

        self.logger.info(
            f"\n=== Journey Summary for {story.id} ===\n"
            f"Movement Metrics:\n"
            f"  Total Distance Traveled: {total_distance:.2f}\n"
            f"  Direct Distance (start to end): {direct_distance:.2f}\n"
            f"  Wandering Ratio: {wandering_ratio:.2f}\n"
            f"\nInteraction Metrics:\n"
            f"  Memories Formed: {len(story.memory_layer)}\n"
            f"  Unique Interactions: {num_unique_interactions}\n"
            f"  Total Perspective Shift: {story.total_perspective_shift:.4f}\n"
            f"  Average Shift Magnitude: {avg_shift:.4f}\n"
            f"  Significant Perspective Changes: {len(significant_shifts)}\n"
            f"\nFinal State:\n"
            f"  Position: {end_state['position']}\n"
            f"  Velocity: {end_state['velocity']}\n"
            f"\nSignificant Events: {len(story.memory_layer)}"
        )


class JourneyLogger:
    def log_interaction(
        self, story1: Story, story2: Story, resonance: float, interaction_type: str
    ):
        latest_memory = story1.memory_layer[-1] if story1.memory_layer else {}
        perspective_shift = latest_memory.get("perspective_shift", 0)

        # If perspective_shift is a coroutine, we need to run it in an event loop
        if asyncio.iscoroutine(perspective_shift):
            perspective_shift = asyncio.get_event_loop().run_until_complete(
                perspective_shift
            )

        log_entry = (
            f"Interaction between {story1.id} and {story2.id}:\n"
            f"  Resonance: {resonance:.2f}\n"
            f"  Interaction Type: {interaction_type}\n"
            f"  Perspective Shift: {perspective_shift:.4f}\n"
            f"  Shared Themes: {set(story1.themes) & set(story2.themes)}\n"
            f"  Distance: {np.linalg.norm(story1.position - story2.position):.2f}\n"
            f"  Positions:\n"
            f"    {story1.id}: {story1.position}\n"
            f"    {story2.id}: {story2.position}\n"
        )
        self.logger.info(log_entry)

    def _format_emotional_change(self, story: Story) -> str:
        return ", ".join(
            [f"{e}: {v:.2f}" for e, v in story.emotional_state.get_dominant_emotions()]
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
        """Enhanced journey summary with accumulated perspective shifts"""
        journey = self.journey_log.get(story.id, [])
        if not journey:
            return

        start_state = journey[0]
        end_state = journey[-1]

        # Calculate metrics
        total_distance = self.total_distances[story.id]
        direct_distance = np.linalg.norm(
            end_state["position"] - start_state["position"]
        )
        wandering_ratio = total_distance / (direct_distance + 1e-6)

        # Perspective analysis
        significant_shifts = [
            s for s in story.perspective_shifts if s["magnitude"] > 0.01
        ]
        avg_shift = (
            np.mean([s["magnitude"] for s in significant_shifts])
            if significant_shifts
            else 0
        )

        # Safely get unique interactions
        unique_interactions = {
            m["interacted_with"]
            for m in story.memory_layer
            if "interacted_with" in m
        }
        num_unique_interactions = len(unique_interactions)

        self.logger.info(
            f"\n=== Journey Summary for {story.id} ===\n"
            f"Movement Metrics:\n"
            f"  Total Distance Traveled: {total_distance:.2f}\n"
            f"  Direct Distance (start to end): {direct_distance:.2f}\n"
            f"  Wandering Ratio: {wandering_ratio:.2f}\n"
            f"\nInteraction Metrics:\n"
            f"  Memories Formed: {len(story.memory_layer)}\n"
            f"  Unique Interactions: {num_unique_interactions}\n"
            f"  Total Perspective Shift: {story.total_perspective_shift:.4f}\n"
            f"  Average Shift Magnitude: {avg_shift:.4f}\n"
            f"  Significant Perspective Changes: {len(significant_shifts)}\n"
            f"\nFinal State:\n"
            f"  Position: {end_state['position']}\n"
            f"  Velocity: {end_state['velocity']}\n"
            f"\nSignificant Events: {len(story.memory_layer)}"
        )


async def create_story_cluster():
    """Create initial story positions in a balanced configuration"""
    # Position stories in a triangle with some random offset
    base_positions = np.array(
        [[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]]
    )

    # Add random offset to make it interesting
    return base_positions + np.random.randn(3, 3) * 0.2


def summarize_story_journey(story: Story):
    """Enhanced journey summary with perspective analysis"""
    theme_counts = {}
    for memory in story.memory_layer:
        for theme in memory["themes"]:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

    most_influential_themes = sorted(
        story.perspective.theme_influences.items(), key=lambda x: x[1], reverse=True
    )[:3]

    total_perspective_shift = story.perspective.total_shift

    return {
        "total_memories": len(story.memory_layer),
        "unique_interactions": len(
            set(m["interacted_with"] for m in story.memory_layer)
        ),
        "theme_exposure": theme_counts,
        "total_perspective_shift": total_perspective_shift,
        "most_influential_themes": most_influential_themes,
        "perspective_shifts": len(
            [s for s in story.perspective.shift_history if s["magnitude"] > 0.01]
        ),
    }


class DynamicThemeGenerator:
    def __init__(self, llm: LanguageModel):
        self.llm = llm
        self.theme_cache = set()
        self.logger = logging.getLogger(__name__)

    async def generate_themes(self, context: str, num_themes: int = 3) -> List[str]:
        prompt = f"""Given the context '{context}', generate {num_themes} unique, single-word themes that could be present in a story. Output ONLY a valid JSON array of strings, nothing else. Example:
        ["hope", "journey", "transformation"]
        """
        response = await self.llm.generate(prompt)
        response = response.strip().lower()

        try:
            # Extract JSON array from the response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                new_themes = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in response")

            # Ensure we have the correct number of themes
            new_themes = [theme.lower() for theme in new_themes[:num_themes]]
            while len(new_themes) < num_themes:
                new_themes.append(f"theme_{len(new_themes) + 1}")

            self.theme_cache.update(new_themes)
            return new_themes
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse themes JSON: {response}")
            return [f"theme_{i+1}" for i in range(num_themes)]
        except Exception as e:
            self.logger.error(f"Error generating themes: {e}")
            return [f"theme_{i+1}" for i in range(num_themes)]

    def get_random_themes(self, num_themes: int = 3) -> List[str]:
        return random.sample(
            list(self.theme_cache), min(num_themes, len(self.theme_cache))
        )


class DynamicStoryGenerator:
    def __init__(self, llm: LanguageModel, theme_generator: DynamicThemeGenerator):
        self.llm = llm
        self.theme_generator = theme_generator
        self.logger = logging.getLogger(__name__)

    async def generate_story(self, field: NarrativeField) -> Story:
        themes = await self.theme_generator.generate_themes("Create a new story")

        prompt = f"Write a short positive, neutral, or negative story (5-8 sentences) incorporating the themes: {', '.join(themes)}. Make it interesting and engaging. Make it personal and emotional. Make it unique and memorable, with one clearly defined protagonist. Start the story by introducing the protagonist's name."
        content = await self.llm.generate(prompt)

        # Extract the protagonist's name from the first sentence
        first_sentence = content.split(".")[0]
        protagonist_name = first_sentence.split()[
            0
        ]  # Assume the first word is the name

        embedding = await self.llm.generate_embedding(content)

        emotional_state = await self.generate_emotional_state(content)

        return Story(
            id=f"story_{len(field.stories)}",
            content=content,
            embedding=np.array(embedding),
            perspective_filter=np.ones(len(embedding)),
            position=np.random.randn(3),
            velocity=np.zeros(3),
            themes=themes,
            emotional_state=emotional_state,
            field=field,
            protagonist_name=protagonist_name,
        )

    async def generate_emotional_state(self, content: str) -> EmotionalState:
        prompt = f"""Given the story: '{content}', describe its positive, neutral, or negative emotional state in 2-3 sentences: """
        description = await self.llm.generate(prompt)
        embedding = await self.llm.generate_embedding(description)
        return EmotionalState(description, np.array(embedding))


class EnvironmentalEventGenerator:
    def __init__(self, llm: LanguageModel):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    async def generate_event(self) -> Dict[str, Any]:
        prompt = "Generate a random positive or negative environmental event for a narrative field. Include an event name, description, and intensity (0.0 to 1.0)."
        response = await self.llm.generate(prompt)

        event_data = {
            "name": "Unknown Event",
            "description": "An unexpected event occurred.",
            "intensity": 0.5,
        }

        for line in response.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ["name", "description"]:
                    event_data[key] = value
                elif key == "intensity":
                    try:
                        event_data[key] = float(value)
                    except ValueError:
                        self.logger.warning(
                            f"Invalid intensity value: {value}. Using default."
                        )

        return event_data

    async def apply_event(self, field: NarrativeField):
        try:
            event = await self.generate_event()
            self.logger.info(
                f"Environmental Event: {event['name']} (Intensity: {event['intensity']})"
            )
            self.logger.info(f"Description: {event['description']}")

            for story in field.stories:
                story.respond_to_environmental_event(event)
        except Exception as e:
            self.logger.error(f"Failed to generate or apply environmental event: {e}")


async def simulate_field():
    """Run a simulation of the narrative field"""
    # Set up logging
    logger = logging.getLogger(__name__)

    # Initialize language model
    llm = OllamaInterface()
    theme_generator = DynamicThemeGenerator(llm)
    story_generator = DynamicStoryGenerator(llm, theme_generator)
    event_generator = EnvironmentalEventGenerator(llm)

    # Initialize simulation components
    field = NarrativeField()
    physics = StoryPhysics()
    visualizer = NarrativeFieldViz()
    collective_engine = EnhancedCollectiveStoryEngine(field)
    interaction_engine = EnhancedInteractionEngine(field, llm)

    # Generate initial stories
    for _ in range(1):
        story = await story_generator.generate_story(field)
        field.add_story(story)

    # Add journey logger
    journey_logger = StoryJourneyLogger()

    # Simulation loop
    for t in range(100):
        field.time = t

        # Update physics
        for story in field.stories:
            physics.update_story_motion(story, field, t)
            journey_logger.log_story_state(story, t)

        # Occasionally generate new stories
        if t % 17 == 0 and len(field.stories) < 10:
            new_story = await story_generator.generate_story(field)
            field.add_story(new_story)
            logger.info(f"New story added: {new_story.id}")

        # Generate and apply environmental events
        if random.random() < 0.025:  # Assuming environmental_event_probability is 0.025
            event = await event_generator.generate_event()
            try:
                await field.apply_environmental_event(event)
                logger.info(
                    f"Environmental Event: {event['name']} (Intensity: {event['intensity']})"
                )
                logger.info(f"Description: {event['description']}")
            except Exception as e:
                logger.error(
                    f"Failed to generate or apply environmental event: {str(e)}"
                )

        # Check for interactions
        for i, story1 in enumerate(field.stories):
            for story2 in field.stories[i + 1 :]:
                distance = np.linalg.norm(story1.position - story2.position)
                if distance < field.interaction_range:
                    resonance, interaction_type = (
                        await interaction_engine.process_interaction(story1, story2)
                    )
                    if resonance > 0:
                        logger.info(f"\nSignificant Interaction:")
                        logger.info(f"  {story1.id} <-> {story2.id}")
                        logger.info(f"  Resonance: {resonance:.2f}")
                        logger.info(
                            f"  Shared Themes: {set(story1.themes) & set(story2.themes)}"
                        )
                        logger.info(f"  Distance: {distance:.2f}")
                        logger.info("  Positions:")
                        logger.info(f"    {story1.id}: {story1.position}")
                        logger.info(f"    {story2.id}: {story2.position}")
                        logger.info(f"\nInteraction Details:")
                        logger.info(f"  Interaction Type: {interaction_type}")
                        logger.info(
                            f"  Perspective Shift: {story1.total_perspective_shift:.4f}"
                        )
                        logger.info(f"\nEmotional Impact:")
                        logger.info(
                            f"  {story1.id} Emotional State: {story1.emotional_state}"
                        )
                        logger.info(
                            f"  {story2.id} Emotional State: {story2.emotional_state}"
                        )

        # Update story states
        await collective_engine.update_story_states()

        # Capture visualization data
        await visualizer.capture_state(field, t)

        # Occasionally generate field pulses and detect patterns
        if t % 30 == 0:
            collective_engine.write_collective_story()
            logger.info(
                f"Collective story at timestep {t}: {collective_engine.collective_story[:500]}..."
            )
            logger.debug(f"Collective story written at timestep {t}")
            logger.debug(f"Collective story: {collective_engine.collective_story}")

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
                    resonance, interaction_type = (
                        await interaction_engine.process_interaction(story1, story2)
                    )
                    journey_logger.log_interaction(
                        story1, story2, resonance, interaction_type
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
