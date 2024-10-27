"""
Narrative Field System

This module implements a framework for analyzing and tracking narrative dynamics in complex social systems.
It uses language models and vector stores to process stories, detect patterns, and analyze the impact of narratives
on a simulated social environment.

Key components:
- Story: Represents individual narratives within the system
- FieldState: Represents the current state of the narrative field
- VectorStore: Abstract base class for storing and retrieving story embeddings
- FieldAnalyzer: Analyzes the impact of stories on the field state
- ResonanceDetector: Finds similarities between stories
- NarrativeField: Main class that orchestrates the entire system
- PerformanceMetrics and PerformanceMonitor: Track and report on system performance
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Final, NewType, Tuple
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import logging.handlers
import psutil
import time
import gc
import atexit
import json
import asyncio
import torch
import chromadb
from chromadb.config import Settings


# Local imports
from logging_config import setup_logging
from language_models import LanguageModel, OllamaInterface, LlamaInterface

# Type Definitions
StoryID = NewType("StoryID", str)

# Constants
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.8
DEFAULT_RESONANCE_LIMIT: Final[int] = 3


class VectorStore(ABC):
    """
    Abstract base class for vector storage and retrieval operations.
    Implementations of this class should provide methods for storing story embeddings
    and finding similar stories based on their embeddings.
    """

    @abstractmethod
    async def store(self, story: Story, embedding: List[float]) -> None:
        """
        Store a story and its embedding in the vector store.

        Args:
            story (Story): The story object to store.
            embedding (List[float]): The embedding vector of the story.
        """
        pass

    @abstractmethod
    async def find_similar(
        self, embedding: List[float], threshold: float, limit: int
    ) -> List[Dict]:
        """
        Find similar stories based on the given embedding.

        Args:
            embedding (List[float]): The query embedding vector.
            threshold (float): The similarity threshold for matching.
            limit (int): The maximum number of similar stories to return.

        Returns:
            List[Dict]: A list of dictionaries containing similar stories and their metadata.
        """
        pass


@dataclass
class Story:
    """
    Represents a single narrative or story within the system.

    Attributes:
        content (str): The main text content of the story.
        context (str): Additional context or metadata for the story.
        id (StoryID): Unique identifier for the story.
        timestamp (datetime): When the story was created or added to the system.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the story.
        resonances (List[str]): IDs of other stories that resonate with this one.
        field_effects (List[Dict]): Analyses of how this story affects the narrative field.
    """

    content: str
    context: str
    id: StoryID = field(default_factory=lambda: StoryID(str(uuid4())))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    resonances: List[str] = field(default_factory=list)
    field_effects: List[Dict] = field(default_factory=list)


@dataclass
class FieldState:
    """
    Represents the current state of the narrative field.

    Attributes:
        description (str): A textual description of the current field state.
        patterns (List[Dict[str, Any]]): Detected patterns in the narrative field.
        active_resonances (List[Dict[str, Any]]): Currently active resonances between stories.
        emergence_points (List[Dict[str, Any]]): Points where new narratives or patterns emerge.
        timestamp (datetime): When this field state was last updated.
    """

    description: str
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    active_resonances: List[Dict[str, Any]] = field(default_factory=list)
    emergence_points: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class FieldAnalysisPrompts:
    """
    A collection of static methods that generate prompts for various field analysis tasks.
    These prompts are used to guide the language model in analyzing the narrative field.
    """

    @staticmethod
    def get_impact_analysis_prompt(story: Story, current_state: FieldState) -> str:
        """
        Generate a prompt for analyzing the impact of a new story on the current field state.

        Args:
            story (Story): The new story to analyze.
            current_state (FieldState): The current state of the narrative field.

        Returns:
            str: A formatted prompt for impact analysis.
        """
        return f"""Analyze how this new narrative affects the existing field state.

Current Field State:
{current_state.description}

New Narrative:
"{story.content}"
Context: {story.context}

Consider and describe:
1. Immediate Effects
- How does this narrative change existing dynamics?
- What emotional responses might emerge?
- Who is most affected and how?

2. Relationship Changes
- How might work relationships shift?
- What new collaborations could form?
- What tensions might develop?

3. Future Implications
- How might this change future interactions?
- What new possibilities emerge?
- What challenges might arise?

Provide a natural, story-focused analysis that emphasizes human impact."""

    @staticmethod
    def get_pattern_detection_prompt(
        stories: List[Story], current_state: FieldState
    ) -> str:
        """
        Generate a prompt for detecting patterns across recent stories.

        Args:
            stories (List[Story]): A list of recent stories to analyze.
            current_state (FieldState): The current state of the narrative field.

        Returns:
            str: A formatted prompt for pattern detection.
        """
        story_summaries = "\n".join(f"- {s.content}" for s in stories[-5:])
        return f"""Analyze patterns and themes across these recent narratives.

Current Field State:
{current_state.description}

Recent Stories:
{story_summaries}

Identify and describe:
1. Emerging Themes
- What recurring topics or concerns appear?
- How are people responding to changes?
- What underlying needs surface?

2. Relationship Patterns
- How are work dynamics evolving?
- What collaboration patterns emerge?
- How is communication changing?

3. Organizational Shifts
- What cultural changes are happening?
- How is the work environment evolving?
- What new needs are emerging?

Describe patterns naturally, focusing on people and relationships."""

    @staticmethod
    def get_resonance_analysis_prompt(story1: Story, story2: Story) -> str:
        """
        Generate a prompt for analyzing the resonance between two stories.

        Args:
            story1 (Story): The first story to compare.
            story2 (Story): The second story to compare.

        Returns:
            str: A formatted prompt for resonance analysis.
        """
        return f"""Analyze how these two narratives connect and influence each other.

First Narrative:
"{story1.content}"
Context: {story1.context}

Second Narrative:
"{story2.content}"
Context: {story2.context}

Examine:
1. Story Connections
- How do these narratives relate?
- What themes connect them?
- How do they influence each other?

2. People Impact
- How might this affect relationships?
- What emotional responses might emerge?
- How might behaviors change?

3. Environment Effects
- How might these stories change the workspace?
- What opportunities might develop?
- What challenges might arise?

Describe connections naturally, focusing on meaning and impact."""


class PerformanceMetrics:
    """
    Tracks and reports performance metrics for various operations in the system.

    This class provides methods to start and stop timers, calculate average durations,
    and log system resource usage.
    """

    def __init__(self):
        """Initialize the PerformanceMetrics object with empty metrics and a logger."""
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    def start_timer(self, operation: str):
        """
        Start a timer for a specific operation.

        Args:
            operation (str): The name of the operation to time.
        """
        if operation not in self.metrics:
            self.metrics[operation] = {
                "start_time": time.perf_counter(),
                "durations": [],
            }
        else:
            self.metrics[operation]["start_time"] = time.perf_counter()

    def stop_timer(self, operation: str) -> float:
        """
        Stop the timer for a specific operation and record its duration.

        Args:
            operation (str): The name of the operation to stop timing.

        Returns:
            float: The duration of the operation in seconds.
        """
        if operation in self.metrics:
            duration = time.perf_counter() - self.metrics[operation]["start_time"]
            self.metrics[operation]["durations"].append(duration)
            return duration
        return 0.0

    def get_average_duration(self, operation: str) -> float:
        """
        Calculate the average duration of a specific operation.

        Args:
            operation (str): The name of the operation to calculate the average for.

        Returns:
            float: The average duration of the operation in seconds.
        """
        if operation in self.metrics and self.metrics[operation]["durations"]:
            return sum(self.metrics[operation]["durations"]) / len(
                self.metrics[operation]["durations"]
            )
        return 0.0

    def print_summary(self):
        """Print a summary of all recorded performance metrics."""
        print("\nPerformance Metrics Summary:")
        for operation, data in self.metrics.items():
            if durations := data["durations"]:
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
                print(f"{operation}:")
                print(f"  Average duration: {avg_duration:.4f} seconds")
                print(f"  Min duration: {min_duration:.4f} seconds")
                print(f"  Max duration: {max_duration:.4f} seconds")
                print(f"  Total calls: {len(durations)}")
            else:
                print(f"{operation}: No data")

    def log_system_resources(self):
        """Log current CPU and memory usage."""
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        self.logger.info(f"CPU Usage: {cpu_percent}%")
        self.logger.info(f"Memory Usage: {memory_info.percent}%")


class PerformanceMonitor:
    """
    Monitors and reports on the performance of language model operations.

    This class provides methods to track generation time and memory usage for LLM operations.
    """

    def __init__(self):
        """Initialize the PerformanceMonitor with an empty list of metrics."""
        self.metrics = []

    async def monitor_generation(
        self, llm: LanguageModel, prompt: str
    ) -> Tuple[str, Dict[str, float]]:
        """
        Monitor the performance of a language model generation task.

        Args:
            llm (LanguageModel): The language model interface to use.
            prompt (str): The prompt to generate a response for.

        Returns:
            Tuple[str, Dict[str, float]]: The generated response and a dictionary of performance metrics.
        """
        start_time = time.perf_counter()
        memory_before = psutil.virtual_memory().used

        response = await llm.generate(prompt)

        end_time = time.perf_counter()
        memory_after = psutil.virtual_memory().used

        metrics = {
            "generation_time": end_time - start_time,
            "memory_usage_change": (memory_after - memory_before) / (1024 * 1024),  # MB
        }

        self.metrics.append(metrics)
        return response, metrics

    def get_performance_report(self) -> Dict[str, float]:
        """
        Generate a report of average performance metrics.

        Returns:
            Dict[str, float]: A dictionary containing average generation time and memory usage change.
        """
        if not self.metrics:
            return {"avg_generation_time": 0, "avg_memory_usage_change": 0}

        return {
            "avg_generation_time": sum(m["generation_time"] for m in self.metrics)
            / len(self.metrics),
            "avg_memory_usage_change": sum(
                m["memory_usage_change"] for m in self.metrics
            )
            / len(self.metrics),
        }


@dataclass
class BatchMetrics:
    """
    Stores metrics for batch processing operations.

    Attributes:
        batch_sizes (List[int]): List of batch sizes processed.
        batch_times (List[float]): List of processing times for each batch.
        memory_usage (List[float]): List of memory usage for each batch.
    """

    batch_sizes: List[int] = field(default_factory=list)
    batch_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)


class BatchProcessor:
    """
    Handles batch processing of prompts using a language model.

    This class optimizes batch sizes based on memory usage and processes prompts in batches.
    """

    def __init__(self, llm: LanguageModel):
        """
        Initialize the BatchProcessor.

        Args:
            llm (LanguageModel): The language model interface to use for processing.
        """
        self.llm = llm
        self.optimal_batch_size = 4  # Will be adjusted dynamically

    async def process_batch(self, prompts: List[str]) -> List[str]:
        """
        Process a batch of prompts using the language model.

        This method dynamically adjusts the batch size based on memory usage and
        processes the prompts in optimal batches.

        Args:
            prompts (List[str]): List of prompts to process.

        Returns:
            List[str]: List of generated responses for each prompt.
        """
        # Dynamic batch size adjustment based on memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_usage > 0.8 * psutil.virtual_memory().total / 1024 / 1024:
            self.optimal_batch_size = max(1, self.optimal_batch_size - 1)

        results = []
        for i in range(0, len(prompts), self.optimal_batch_size):
            batch = prompts[i : i + self.optimal_batch_size]
            batch_results = await asyncio.gather(
                *[self.llm.generate(prompt) for prompt in batch]
            )
            results.extend(batch_results)

        return results


class ChromaStore(VectorStore):
    """
    Implementation of VectorStore using Chroma DB for storing and retrieving story embeddings.

    This class provides methods to store story embeddings and find similar stories based on
    cosine similarity of their embeddings.
    """

    def __init__(self, collection_name: str = "narrative_field"):
        """
        Initialize the ChromaStore.

        Args:
            collection_name (str): Name of the Chroma DB collection to use.
        """
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.logger = logging.getLogger(__name__)

        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

    async def store(self, story: Story, embedding: List[float]) -> None:
        """
        Store a story and its embedding in the Chroma DB.

        Args:
            story (Story): The story object to store.
            embedding (List[float]): The embedding vector of the story.
        """
        metadata = {
            "content": story.content,
            "context": story.context,
            "timestamp": story.timestamp.isoformat(),
            "resonances": json.dumps(story.resonances),
            "field_effects": json.dumps(
                [
                    {
                        "analysis": effect["analysis"],
                        "timestamp": effect["timestamp"].isoformat(),
                        "story_id": effect["story_id"],
                    }
                    for effect in story.field_effects
                ]
            ),
        }

        await asyncio.to_thread(
            self.collection.add,
            documents=[json.dumps(metadata)],
            embeddings=[embedding],
            ids=[story.id],
            metadatas=[metadata],
        )

    async def find_similar(
        self,
        embedding: List[float],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        limit: int = DEFAULT_RESONANCE_LIMIT,
    ) -> List[Dict]:
        """
        Find similar stories based on the given embedding.

        Args:
            embedding (List[float]): The query embedding vector.
            threshold (float): The similarity threshold for matching.
            limit (int): The maximum number of similar stories to return.

        Returns:
            List[Dict]: A list of dictionaries containing similar stories and their metadata.
        """
        count = self.collection.count()
        if count == 0:
            return []

        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[embedding],
            n_results=min(limit, count),
        )

        similar = []
        for idx, id in enumerate(results["ids"][0]):
            metadata = json.loads(results["documents"][0][idx])
            similar.append(
                {
                    "id": id,
                    "similarity": results["distances"][0][idx],
                    "metadata": metadata,
                }
            )

        return [s for s in similar if s["similarity"] <= threshold]


class FieldAnalyzer:
    """
    Analyzes the impact of stories on the narrative field and detects patterns.

    This class uses a language model to generate analyses based on prompts.
    """

    def __init__(self, llm_interface: LanguageModel):
        """
        Initialize the FieldAnalyzer.

        Args:
            llm_interface (LanguageModel): The language model interface to use for analysis.
        """
        self.llm = llm_interface
        self.logger = logging.getLogger(__name__)
        self.prompts = FieldAnalysisPrompts()

    async def analyze_impact(
        self, story: Story, current_state: FieldState
    ) -> Dict[str, Any]:
        """
        Analyze the impact of a new story on the current field state.

        Args:
            story (Story): The new story to analyze.
            current_state (FieldState): The current state of the narrative field.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis, timestamp, and story ID.
        """
        prompt = self.prompts.get_impact_analysis_prompt(story, current_state)
        analysis = await self.llm.generate(prompt)

        return {"analysis": analysis, "timestamp": datetime.now(), "story_id": story.id}

    async def detect_patterns(
        self, stories: List[Story], current_state: FieldState
    ) -> str:
        """
        Detect patterns across a list of stories in the context of the current field state.

        Args:
            stories (List[Story]): The list of stories to analyze for patterns.
            current_state (FieldState): The current state of the narrative field.

        Returns:
            str: A string describing the detected patterns.
        """
        prompt = self.prompts.get_pattern_detection_prompt(stories, current_state)
        return await self.llm.generate(prompt)


class ResonanceDetector:
    """
    Detects resonances between stories in the narrative field.

    This class uses a vector store to find similar stories and a language model
    to analyze the nature of the resonance between stories.
    """

    def __init__(self, vector_store: VectorStore, llm_interface: LanguageModel):
        """
        Initialize the ResonanceDetector.

        Args:
            vector_store (VectorStore): The vector store to use for finding similar stories.
            llm_interface (LanguageModel): The language model interface to use for analysis.
        """
        self.vector_store = vector_store
        self.llm = llm_interface
        self.logger = logging.getLogger(__name__)
        self.prompts = FieldAnalysisPrompts()

    async def find_resonances(
        self,
        story: Story,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        limit: int = DEFAULT_RESONANCE_LIMIT,
    ) -> List[Dict[str, Any]]:
        """
        Find resonances between a given story and other stories in the vector store.

        Args:
            story (Story): The story to find resonances for.
            threshold (float): The similarity threshold for considering a resonance.
            limit (int): The maximum number of resonances to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries describing the resonances found.
        """
        try:
            self.logger.debug(f"Generating embedding for story: {story.id}")
            # Ensure embedding is generated before using it
            embedding = await self.llm.generate_embedding(
                f"{story.content} {story.context}"
            )
            similar_stories = await self.vector_store.find_similar(
                embedding, threshold, limit
            )
            self.logger.debug(f"Found {len(similar_stories)} similar stories")

            resonances = []
            for similar in similar_stories:
                metadata = similar["metadata"]
                similar_story = Story(
                    id=similar["id"],
                    content=metadata["content"],
                    context=metadata["context"],
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                )

                resonance = await self.determine_resonance_type(story, similar_story)
                resonances.append(
                    {
                        "story_id": similar["id"],
                        "resonance": resonance,
                        "timestamp": datetime.now(),
                    }
                )

            self.logger.debug(f"Generated {len(resonances)} resonances")
            return resonances
        except Exception as e:
            self.logger.error(f"Error in find_resonances: {e}", exc_info=True)
            raise

    async def determine_resonance_type(
        self, story1: Story, story2: Story
    ) -> Dict[str, Any]:
        """
        Determine the type and nature of resonance between two stories.

        Args:
            story1 (Story): The first story in the resonance pair.
            story2 (Story): The second story in the resonance pair.

        Returns:
            Dict[str, Any]: A dictionary describing the resonance between the stories.
        """
        prompt = self.prompts.get_resonance_analysis_prompt(story1, story2)
        analysis = await self.llm.generate(prompt)

        return {
            "type": "narrative_resonance",
            "analysis": analysis,
            "stories": {
                "source": {
                    "id": story1.id,
                    "content": story1.content,
                    "context": story1.context,
                },
                "resonant": {
                    "id": story2.id,
                    "content": story2.content,
                    "context": story2.context,
                },
            },
            "timestamp": datetime.now(),
        }


class NarrativeField:
    """
    Main class that orchestrates the entire Narrative Field System.

    This class manages the addition of new stories, updates to the field state,
    and coordinates the analysis and resonance detection processes.
    """

    def __init__(self, llm_interface: LanguageModel, vector_store: VectorStore):
        """
        Initialize the NarrativeField.

        Args:
            llm_interface (LanguageModel): The language model interface to use for analysis.
            vector_store (VectorStore): The vector store to use for story storage and retrieval.
        """
        self._analyzer = FieldAnalyzer(llm_interface)
        self._resonance_detector = ResonanceDetector(vector_store, llm_interface)
        self._vector_store = vector_store
        self._state = FieldState(description="Initial empty narrative field")
        self._stories: Dict[StoryID, Story] = {}
        self._logger = logging.getLogger(__name__)
        self._performance_metrics = PerformanceMetrics()

    @property
    def state(self) -> FieldState:
        """Get the current state of the narrative field."""
        return self._state

    @property
    def stories(self) -> Dict[StoryID, Story]:
        """Get a copy of all stories in the narrative field."""
        return self._stories.copy()

    async def add_story(self, content: str, context: str) -> Story:
        """
        Add a new story to the narrative field and analyze its impact.

        This method creates a new story, analyzes its impact on the field,
        finds resonances with other stories, and updates the field state.

        Args:
            content (str): The main content of the story.
            context (str): The context or additional information about the story.

        Returns:
            Story: The newly created and analyzed story object.
        """
        self._performance_metrics.start_timer("add_story")

        self._performance_metrics.start_timer("create_story")
        story = Story(content=content, context=context)
        create_time = self._performance_metrics.stop_timer("create_story")
        self._logger.info(f"Story creation time: {create_time:.4f} seconds")

        self._performance_metrics.start_timer("analyze_impact")
        impact = await self._analyzer.analyze_impact(story, self.state)
        analyze_time = self._performance_metrics.stop_timer("analyze_impact")
        self._logger.info(f"Impact analysis time: {analyze_time:.4f} seconds")
        story.field_effects.append(impact)

        self._performance_metrics.start_timer("find_resonances")
        resonances = await self._resonance_detector.find_resonances(story)
        resonance_time = self._performance_metrics.stop_timer("find_resonances")
        self._logger.info(f"Find resonances time: {resonance_time:.4f} seconds")
        story.resonances.extend([r["story_id"] for r in resonances])

        self._performance_metrics.start_timer("store_story")
        await self._store_story(story)
        store_time = self._performance_metrics.stop_timer("store_story")
        self._logger.info(f"Store story time: {store_time:.4f} seconds")

        self._performance_metrics.start_timer("update_field_state")
        await self._update_field_state(story, impact, resonances)
        update_time = self._performance_metrics.stop_timer("update_field_state")
        self._logger.info(f"Update field state time: {update_time:.4f} seconds")

        total_time = self._performance_metrics.stop_timer("add_story")
        self._logger.info(f"Total add_story time: {total_time:.4f} seconds")

        self._performance_metrics.log_system_resources()

        return story

    async def _store_story(self, story: Story) -> None:
        """
        Store a story in the vector store and local dictionary.

        Args:
            story (Story): The story to store.
        """
        embedding = await self._resonance_detector.llm.generate_embedding(
            f"{story.content} {story.context}"
        )
        await self._vector_store.store(story, embedding)
        self._stories[story.id] = story

    async def _update_field_state(
        self, story: Story, impact: Dict, resonances: List[Dict]
    ) -> None:
        """
        Update the field state based on a new story, its impact, and resonances.

        Args:
            story (Story): The new story added to the field.
            impact (Dict): The impact analysis of the story.
            resonances (List[Dict]): The resonances found for the story.
        """
        patterns = await self._analyzer.detect_patterns(
            list(self._stories.values()), self.state
        )

        self._state = FieldState(
            description=impact["analysis"],
            patterns=[{"analysis": patterns}],
            active_resonances=resonances,
            emergence_points=[
                {
                    "story_id": story.id,
                    "timestamp": datetime.now(),
                    "type": "new_narrative",
                    "resonance_context": [
                        r["resonance"]["analysis"] for r in resonances
                    ],
                }
            ],
        )


# Global cleanup function
def global_cleanup():
    """
    Perform global cleanup operations.

    This function is called at program exit to free up resources and clear caches.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Register the global cleanup function to run at exit
atexit.register(global_cleanup)


async def demo_scenario():
    """
    Run a demonstration scenario of the Narrative Field System.

    This function sets up the system, processes a series of stories,
    and logs performance metrics throughout the process.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting narrative field demonstration...")

    # Initialize performance monitor
    monitor = PerformanceMonitor()

    llm = None  # Initialize llm to None

    try:
        # Perform global cleanup before initializing new LLM
        global_cleanup()

        llm = LlamaInterface()

        vector_store: VectorStore = ChromaStore(collection_name="research_lab")
        logger.info("Initialized Chroma vector store")

        field = NarrativeField(llm, vector_store)
        logger.info("Initialized narrative field")

        # Research Lab Scenario with Multiple Characters and events
        stories = [
            # Event 1: Leon discussing the AI minor
            {
                "content": "After lunch, as Leon and Coen walked back to the lab, Leon decided to share his growing concerns about the AI for Society minor. He voiced his doubts and the challenges he foresaw in the program's current direction. Coen listened attentively and was supportive of Leon's worries. \"I think you have some valid points,\" Coen acknowledged, \"but perhaps it would be best to discuss these issues with Danny, the manager of the minor.\" Coen believed that Danny's insights could be crucial in addressing Leon's concerns.",
                "context": "Leon confides in Coen about issues with the AI minor; Coen advises consulting Danny.",
            },
            # Event 2: Robbert's tough advice
            {
                "content": "After work, Robbert and Leon walked back to the lab together. Leon expressed his worries about Danny's accident and the AI minor. However, Robbert seemed more preoccupied with his own research and was not interested in discussing the minor. \"I know you're concerned, but you need to man up and stop whining,\" Robbert said bluntly. His tough advice left Leon feeling isolated.",
                "context": "Robbert dismisses Leon's concerns, focusing instead on his own research priorities.",
            },
            # Event 4: Sarah's contribution
            {
                "content": "Sarah, a new member of the lab eager to make her mark, approached Leon with a fresh idea. Enthusiastic about the ethical challenges in AI, she suggested a new direction for the AI minor—focusing on ethics in AI development. Her excitement was contagious, and Leon began to see the potential impact of integrating ethics into the program.",
                "context": "Sarah proposes refocusing the AI minor on AI ethics, sparking interest from Leon.",
            },
            # Event 5: Tom's exhaustion
            {
                "content": "Tom, another member of the lab, was visibly exhausted after a long day. He had been struggling to keep up with the heavy workload and confided in his colleagues that he wanted to leave early. Considering taking a break from the lab altogether, Tom felt mentally drained and knew he needed time to recover.",
                "context": "Tom is overwhelmed by work stress and thinks about temporarily leaving the lab.",
            },
            # Event 6: Leon reassessing
            {
                "content": "Observing Tom's exhaustion, Leon became concerned that the lab might be overworking its members. Balancing his worries about the AI minor and the well-being of his colleagues, he suggested organizing a team meeting to discuss workload management. Leon hoped that addressing these issues openly would help prevent burnout and improve overall productivity.",
                "context": "Leon considers holding a meeting to tackle workload issues affecting team morale.",
            },
            # Event 7: Coen's personal struggle
            {
                "content": "In a candid conversation, Coen revealed to Leon that he had been dealing with personal issues and was struggling to focus on work. Leon was surprised by Coen's admission, as he had always appeared to have everything under control. This revelation highlighted the underlying stress affecting the team.",
                "context": "Coen admits personal struggles are hindering his work, surprising Leon.",
            },
            # Event 8: Sarah's proposal
            {
                "content": "Concerned about her colleagues' mental health, Sarah proposed implementing a flexible working schedule to accommodate those feeling burned out. She believed that a healthier work-life balance would benefit both the individuals and the lab's productivity. \"We need to take care of ourselves to do our best work,\" she advocated.",
                "context": "Sarah suggests flexible hours to improve well-being and efficiency in the lab.",
            },
            # Event 9: Tom's decision
            {
                "content": "Feeling overwhelmed, Tom decided to take a temporary leave from the lab to focus on his mental health. He believed that stepping back was the best decision for now and hoped that his absence would prompt the team to consider the pressures they were all facing.",
                "context": "Tom takes a break to address his mental health, hoping to highlight team stress.",
            },
            # Event 10: Sarah's pushback
            {
                "content": "Sarah pushed back against Robbert's position during the meeting, arguing that a more flexible approach would ultimately lead to better results. She highlighted the risks of burnout and the benefits of supporting team members through their personal struggles. The team found itself divided between Robbert's hardline approach and Sarah's call for change.",
                "context": "Sarah challenges Robbert's views, leading to a team split over work policies.",
            },
            # Event 11: A breakthrough idea
            {
                "content": "During a late-night discussion, Leon and Sarah brainstormed a novel approach to restructure the AI minor. They envisioned incorporating elements of ethics and mental health awareness into the curriculum, aligning the program with current societal needs. Energized by this new direction, Leon believed it could address both the challenges facing the AI minor and the lab's workload issues.",
                "context": "Leon and Sarah create a plan integrating ethics and mental health into the AI minor.",
            },
            # Event 12: Tom's return
            {
                "content": "After his break, Tom returned to the lab feeling refreshed and ready to contribute again. He appreciated the support from his colleagues and felt more optimistic about balancing his mental health with work. Tom's return brought a renewed sense of hope to the team, signaling the potential for positive change.",
                "context": "Tom's rejuvenated return inspires hope for better balance in the lab.",
            },
        ]

        # Process stories with performance monitoring
        logger.info(f"Processing {len(stories)} stories and analyzing field effects...")
        for story in stories:
            try:
                metrics = await monitor.monitor_generation(llm, story["content"])
                logger.debug(f"Story processing metrics: {metrics}")

                await field.add_story(story["content"], story["context"])

            except Exception as e:
                logger.error(f"Error processing story: {e}", exc_info=True)
                continue

        # Log performance report at the end
        performance_report = monitor.get_performance_report()
        logger.info(f"Performance Report: {performance_report}")

        # Print the detailed performance metrics summary
        field._performance_metrics.print_summary()

    except Exception as e:
        logger.error(f"Error in demo scenario: {e}", exc_info=True)
        raise
    finally:
        # Clean up resources
        if llm is not None:
            await llm.cleanup()
        global_cleanup()
        logger.info("Narrative field demonstration completed")


if __name__ == "__main__":
    try:
        setup_logging()  # Call the setup_logging function from the imported module
        asyncio.run(demo_scenario())
    finally:
        global_cleanup()
