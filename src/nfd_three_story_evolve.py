import sys
from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from language_models import LanguageModel, OllamaInterface
import asyncio


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


class NarrativeField:
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.stories: List[Story] = []
        self.field_memory = []
        self.collective_state = np.zeros(dimension)
        self.time = 0.0

        # Initialize field properties
        self.field_potential = np.zeros(dimension)
        self.resonance_threshold = 0.7

        self.logger = logging.getLogger(__name__)

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

        return float(
            0.4 * embedding_similarity + 0.3 * theme_overlap + 0.3 * filter_alignment
        )


class StoryInteractionEngine:
    """Handles interactions between stories in the field"""

    def __init__(self, field: NarrativeField):
        self.field = field
        self.logger = logging.getLogger(__name__)

    def process_interaction(self, story1: Story, story2: Story):
        """Process an interaction between two stories"""
        resonance = self.field.detect_resonance(story1, story2)

        if resonance > self.field.resonance_threshold:
            self.logger.info(
                f"Interaction between {story1.id} and {story2.id} with resonance {resonance:.2f}"
            )

            # Create memory imprints
            memory1 = {
                "time": self.field.time,
                "interacted_with": story2.id,
                "resonance": resonance,
                "themes": story2.themes,
            }
            memory2 = {
                "time": self.field.time,
                "interacted_with": story1.id,
                "resonance": resonance,
                "themes": story1.themes,
            }

            # Update memory layers
            story1.memory_layer.append(memory1)
            story2.memory_layer.append(memory2)

            # Update perspective filters
            self._update_perspective_filter(story1, story2)
            self._update_perspective_filter(story2, story1)

    def _update_perspective_filter(self, story1: Story, story2: Story):
        """Update a story's perspective filter based on interaction"""
        # Blend perspective filters with a decay factor
        decay = 0.9
        story1.perspective_filter = (
            decay * story1.perspective_filter + (1 - decay) * story2.perspective_filter
        )


class CollectiveStoryEngine:
    """Manages collective field effects and emergent patterns"""

    def __init__(self, field: NarrativeField):
        self.field = field
        self.collective_memories = []
        self.logger = logging.getLogger(__name__)

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

    def detect_emergent_patterns(self) -> List[Dict]:
        """Detect emergent patterns in the field"""
        self.logger.info("Detecting emergent patterns in the field")

        # Analyze recent interactions and memory patterns
        interaction_threshold = 0.5
        recent_interactions = [
            {
                "story_id": story.id,
                "interacted_with": memory["interacted_with"],
                "themes": memory["themes"],
                "resonance": memory["resonance"],
            }
            for story in self.field.stories
            for memory in story.memory_layer[-10:]
            if memory["resonance"] > interaction_threshold
        ]

        # Count theme occurrences
        theme_counts = {}
        for interaction in recent_interactions:
            for theme in interaction["themes"]:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Identify most common themes
        common_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]

        # Analyze resonance patterns
        avg_resonance = (
            sum(inter["resonance"] for inter in recent_interactions)
            / len(recent_interactions)
            if recent_interactions
            else 0
        )

        self.logger.info(
            f"Detected {len(recent_interactions)} significant recent interactions"
        )
        self.logger.info(f"Most common themes: {common_themes}")
        self.logger.info(f"Average resonance: {avg_resonance:.2f}")

        # Add detected patterns to the patterns list
        return [
            {
                "type": "recent_interactions",
                "count": len(recent_interactions),
                "common_themes": common_themes,
                "avg_resonance": avg_resonance,
            }
        ]


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
        position=np.zeros(3),
        velocity=np.zeros(3),
        themes=["loneliness", "duty", "hope", "guidance"],
        resonance_history=[],
    )


async def create_path_story(llm: LanguageModel) -> Story:
    content = """
    The Path Through the Forest
    This story is about a traveler lost in a forest, searching for a way home. It holds elements of uncertainty, hope, and resilience as the traveler navigates unfamiliar terrain, feeling both wonder and isolation. 
    """
    embedding = await llm.generate_embedding(content)
    return Story(
        id="path",
        content=content,
        embedding=np.array(embedding),
        memory_layer=[],
        perspective_filter=np.ones(len(embedding)),
        position=np.random.randn(3),
        velocity=np.zeros(3),
        themes=["journey", "discovery", "nature"],
        resonance_history=[],
    )


async def create_dream_story(llm: LanguageModel) -> Story:
    content = """
    The Child's Dream of Flight
    This story follows a child who dreams each night of flying, soaring above villages, forests, and oceans. The dream is filled with freedom, innocence, and limitless possibility, untouched by fear or doubt.
    """
    embedding = await llm.generate_embedding(content)
    return Story(
        id="dream",
        content=content,
        embedding=np.array(embedding),
        memory_layer=[],
        perspective_filter=np.ones(len(embedding)),
        position=np.random.randn(3),
        velocity=np.zeros(3),
        themes=["imagination", "freedom", "subconscious"],
        resonance_history=[],
    )


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

    # Initialize field
    field = NarrativeField()
    interaction_engine = StoryInteractionEngine(field)
    collective_engine = CollectiveStoryEngine(field)

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

    # Simulation loop
    for t in range(1000):  # 1000 timesteps
        field.time = t
        logger.debug(f"Simulation timestep: {t}")

        # Update story positions
        for story in field.stories:
            story.position += story.velocity

        # Check for interactions
        for i, story1 in enumerate(field.stories):
            for story2 in field.stories[i + 1 :]:
                if np.linalg.norm(story1.position - story2.position) < 1.0:
                    interaction_engine.process_interaction(story1, story2)

        # Occasionally generate field pulses
        if t % 100 == 0:
            collective_engine.generate_field_pulse("seeking_connection", 0.5)

        # Update field state
        field._update_field_potential()

    logger.info("Narrative field simulation completed")

    # Clean up
    await llm.cleanup()


if __name__ == "__main__":
    asyncio.run(simulate_field())


if __name__ == "__main__":
    asyncio.run(simulate_field())
