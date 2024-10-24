from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Dict

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, TypedDict, Union, Literal, NewType, Optional, Final
from uuid import uuid4
from enum import Enum, auto
from typing_extensions import TypeGuard
import typing

import chromadb
import ollama
from chromadb.config import Settings

DEFAULT_SIMILARITY_THRESHOLD: Final[float] = typing.cast(float, 0.8)
DEFAULT_RESONANCE_LIMIT: Final[int] = typing.cast(int, 3)
MAX_LOG_FILE_SIZE: Final[int] = typing.cast(int, 10 * 1024 * 1024)  # 10 MB
MAX_LOG_BACKUP_COUNT: Final[int] = typing.cast(int, 19)  # 20 files total

# Create a logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log file name based on the current timestamp
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"nfs_lab_scenario_{current_time}.log")

# Set up the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=MAX_LOG_FILE_SIZE,
    backupCount=MAX_LOG_BACKUP_COUNT,
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

StoryID = NewType("StoryID", str)


# 1. Define base classes and abstract classes
class VectorStore(ABC):
    @abstractmethod
    async def store(self, story: "Story", embedding: List[float]) -> None:
        pass

    @abstractmethod
    async def find_similar(
        self, embedding: List[float], threshold: float, limit: int
    ) -> List[Dict]:
        pass


class LanguageModel(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        pass


# 2. Define data classes and enums
@dataclass
class Story:
    content: str
    context: str
    id: StoryID = field(default_factory=lambda: StoryID(str(uuid4())))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = field(default=None)
    resonances: List[str] = field(default_factory=list)
    field_effects: List[Dict] = field(default_factory=list)


@dataclass
class Pattern:
    name: str
    description: str
    strength: float = 0.0
    related_stories: List[StoryID] = field(default_factory=list)

    def update_strength(self, new_strength: float):
        self.strength = new_strength

    def add_related_story(self, story_id: StoryID):
        if story_id not in self.related_stories:
            self.related_stories.append(story_id)


@dataclass
class Resonance:
    type: str
    strength: float
    source_story: StoryID
    target_story: StoryID
    description: str = ""

    def update_strength(self, new_strength: float):
        self.strength = new_strength


@dataclass
class EmergencePoint:
    story_id: StoryID
    timestamp: datetime
    type: str = "new_narrative"
    resonance_context: List[Resonance] = field(default_factory=list)
    related_patterns: List[Pattern] = field(default_factory=list)

    def add_related_pattern(self, pattern: Pattern):
        if pattern not in self.related_patterns:
            self.related_patterns.append(pattern)


@dataclass
class FieldState:
    description: str
    patterns: List[Pattern] = field(default_factory=list)
    active_resonances: List[Resonance] = field(default_factory=list)
    emergence_points: List[EmergencePoint] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_pattern(self, pattern: Pattern):
        self.patterns.append(pattern)

    def add_resonance(self, resonance: Resonance):
        self.active_resonances.append(resonance)

    def add_emergence_point(self, point: EmergencePoint):
        self.emergence_points.append(point)

    def update_description(self, new_description: str):
        self.description = new_description
        self.timestamp = datetime.now()


class ImpactAnalysis(TypedDict):
    analysis: str
    timestamp: datetime
    story_id: str


class ResonanceAnalysis(TypedDict):
    type: Literal["narrative_resonance"]
    analysis: str
    stories: Dict[
        Literal["source", "resonant"], Dict[Literal["id", "content", "context"], str]
    ]
    timestamp: datetime


# 3. Implement concrete classes
class OllamaInterface(LanguageModel):
    def __init__(
        self,
        model_name: str = "mistral-nemo",
        embed_model_name: str = "mxbai-embed-large",
    ):
        self.model: str = model_name
        self.embed_model: str = embed_model_name
        self.embedding_cache: Dict[str, List[float]] = {}
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initializing OllamaInterface with models: main={self.model}, embedding={self.embed_model}"
        )

    async def generate(self, prompt: str) -> str:
        self.logger.debug(f"Sending prompt to LLM: {prompt}")
        self.logger.info(f"Prompt length sent to LLM: {len(prompt)} characters")
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"]
        self.logger.debug(f"Received response from LLM: {content}")
        self.logger.info(f"Response length from LLM: {len(content)} characters")
        return content

    async def generate_embedding(self, story: Story) -> List[float]:
        # Use the Story's GUID as the cache key
        cache_key = story.id
        text_to_embed = f"{story.content} {story.context}"

        # Check cache first
        self.logger.info("Checking embedding cache before generating")
        if cache_key in self.embedding_cache:
            self.logger.info(
                f"Embedding retrieved from cache for story ID: {cache_key}"
            )
            return self.embedding_cache[cache_key]

        # Generate if not cached
        response = await asyncio.to_thread(
            ollama.embeddings, model=self.embed_model, prompt=text_to_embed
        )
        embedding = response["embedding"]

        # Cache the result
        self.embedding_cache[cache_key] = embedding
        self.logger.info(
            f"Embedding generated and cached successfully for story ID: {cache_key} with length {len(embedding)}"
        )
        return embedding


class ChromaStore(VectorStore):
    def __init__(self, collection_name: str = "narrative_field"):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.logger = logging.getLogger(__name__)
        try:
            self.collection = self.client.get_collection(collection_name)
            self.logger.info(f"Collection {collection_name} found")
            self.logger.info(f"Collection metadata: {self.collection.metadata}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Collection {collection_name} created")
            self.logger.info(f"Collection metadata: {self.collection.metadata}")

    async def store(self, story: Story, embedding: List[float]) -> None:
        """Store story embedding and metadata"""
        self.logger.info(f"Storing embedding and metadata for story: {story.id}")
        self.logger.debug(f"Embedding length: {len(embedding)}")

        metadata = {
            "content": story.content,
            "context": story.context,
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
            "resonances": json.dumps(story.resonances),
            "timestamp": story.timestamp.isoformat(),
        }

        await asyncio.to_thread(
            self.collection.add,
            documents=[json.dumps(metadata)],
            embeddings=[embedding],
            ids=[story.id],
            metadatas=[metadata],
        )

    async def find_similar(
        self, embedding: List[float], threshold: float = 0.8, limit: int = 5
    ) -> List[Dict]:
        """Find similar narratives"""
        self.logger.info(
            f"Finding similar narratives with threshold: {threshold} and limit: {limit}"
        )
        count = self.collection.count()
        if count == 0:
            return []

        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[embedding],
            n_results=min(limit, count),
        )

        self.logger.info(f"Found {len(results['ids'][0])} similar narratives")
        self.logger.debug(f"Similar narratives results: {results}")

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

        thresholded = [s for s in similar if s["similarity"] <= threshold]
        self.logger.info(
            f"Thresholded results length similarity narrative: {len(thresholded)}"
        )
        self.logger.debug(f"Thresholded results similarity narrative: {thresholded}")

        return thresholded


# 4. Define main logic classes
class FieldAnalyzer:
    def __init__(self, llm_interface: LanguageModel):
        self.llm: LanguageModel = llm_interface
        self.logger: logging.Logger = logging.getLogger(__name__)

    async def analyze_impact(
        self, story: Story, current_state: FieldState
    ) -> Dict[str, Any]:
        """Analyze how a story impacts the field"""
        prompt = f"""
        Current field state: {current_state.description}
        Active patterns: {current_state.patterns}
        
        New narrative entering field:
        Content: {story.content}
        Context: {story.context}
        
        Analyze field impact:
        1. Immediate resonance effects
        2. Pattern interactions/disruptions
        3. Potential emergence points
        4. Field state transformations
        5. Emotional impact on the field
        6. Narrative evolution
        
        Provide a qualitative, story-driven analysis without using numeric measures.
        """

        analysis = await self.llm.generate(prompt)

        result = {
            "analysis": analysis,
            "timestamp": datetime.now(),
            "story_id": story.id,
        }
        return result

    async def detect_patterns(
        self, stories: List[Story], current_state: FieldState
    ) -> List[Dict[str, Any]]:
        """Identify emergent patterns in the narrative field"""
        self.logger.info(f"Detecting patterns for {len(stories)} stories")
        self.logger.debug(f"Current field state: {current_state.description}")
        self.logger.info(
            f"Current field state length: {len(current_state.description)}"
        )

        story_contexts = [
            {"content": s.content, "context": s.context, "effects": s.field_effects}
            for s in stories
        ]

        prompt = f"""
        Analyze narrative collection for emergent patterns:
        Stories: {story_contexts}
        Current Patterns: {current_state.patterns}
        Active Resonances: {current_state.active_resonances}
        
        Identify:
        1. New pattern formation
        2. Pattern evolution/dissolution
        3. Resonance networks
        4. Critical transition points
        5. Emergence phenomena
        
        Use the Stories, Current Patterns and Active Resonances to determine the impact. NO markdown or code blocks.
        """

        self.logger.debug(
            f"Sending prompt to LLM for emergent pattern detection: {prompt}"
        )
        self.logger.info(
            f"Emergent pattern detection prompt length: {len(prompt)} characters"
        )

        patterns = await self.llm.generate(prompt)
        self.logger.debug(f"Received emergent patterns response: {patterns}")
        self.logger.info(
            f"Emergent pattern detection response length: {len(patterns)} characters"
        )
        return patterns


class ResonanceDetector:
    def __init__(self, vector_store: VectorStore, llm_interface: LanguageModel):
        self.vector_store: VectorStore = vector_store
        self.llm: LanguageModel = llm_interface
        self.logger: logging.Logger = logging.getLogger(__name__)

    async def find_resonances(
        self,
        story: Story,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        limit: int = DEFAULT_RESONANCE_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Find and analyze resonating stories using semantic understanding"""
        self.logger.debug(f"Finding resonances for story: {story.id}")

        embedding = await self.llm.generate_embedding(story)
        self.logger.debug(f"Generated embedding for story: {story.id}")

        similar_stories = await self.vector_store.find_similar(
            embedding, threshold=threshold, limit=limit
        )
        self.logger.debug(f"Found {len(similar_stories)} similar stories")

        resonances = []
        for similar in similar_stories:
            self.logger.debug(f"Analyzing resonance with story: {similar['id']}")
            similar_metadata = similar["metadata"]
            similar_story = Story(
                id=similar["id"],
                content=similar_metadata["content"],
                context=similar_metadata["context"],
                timestamp=(
                    datetime.fromisoformat(similar_metadata["timestamp"])
                    if isinstance(similar_metadata["timestamp"], str)
                    else similar_metadata["timestamp"]
                ),
            )

            resonance = await self.determine_resonance_type(story, similar_story)
            resonances.append(
                {
                    "story_id": similar["id"],
                    "resonance": resonance,
                    "timestamp": datetime.now(),
                }
            )
            self.logger.info(f"Resonance analysis completed for story: {similar['id']}")

        self.logger.info(f"Found {len(resonances)} resonances for story: {story.id}")
        return resonances

    async def determine_resonance_type(
        self, story1: Story, story2: Story
    ) -> ResonanceAnalysis:
        prompt = f"""
        Analyze the narrative resonance between these two stories:
        
        Story 1: {story1.content}
        Context 1: {story1.context}
        
        Story 2: {story2.content}
        Context 2: {story2.context}
        
        Provide a detailed analysis:
        1. Narrative Relationship:
           - How do these stories interact on a narrative level?
           - What kind of thematic or emotional connection exists?
           - How do they reinforce, conflict with, or transform each other's meanings?
        
        2. Character Development:
           - How might these stories influence the characters' growth or change?
           - What new aspects of personality or motivation might emerge?
        
        3. Worldview Impact:
           - How do these stories shape the characters' understanding of their world?
           - What beliefs or values are being challenged or reinforced?
        
        Provide a qualitative, story-driven analysis without using numeric measures.
        """

        analysis = await self.llm.generate(prompt)

        result: ResonanceAnalysis = {
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
        return result


class NarrativeField:
    def __init__(self, llm_interface: LanguageModel, vector_store: VectorStore):
        self._analyzer: FieldAnalyzer = FieldAnalyzer(llm_interface)
        self._resonance_detector: ResonanceDetector = ResonanceDetector(
            vector_store, llm_interface
        )
        self._vector_store: VectorStore = vector_store
        self._state: FieldState = FieldState(
            description="Initial empty narrative field"
        )
        self._stories: Dict[StoryID, Story] = {}
        self._logger: logging.Logger = logging.getLogger(__name__)

    @property
    def state(self) -> FieldState:
        return self._state

    @property
    def stories(self) -> Dict[StoryID, Story]:
        return self._stories.copy()  # Return a copy to prevent direct modification

    async def add_story(self, content: str, context: str) -> Story:
        story: Story = Story(content=content, context=context)
        self._logger.info(f"Adding new story: {story.id}")
        self._logger.debug(f"Story content: {story.content}")
        self._logger.debug(f"Story context: {story.context}")

        # Analyze field impact
        impact: ImpactAnalysis = await self._analyzer.analyze_impact(story, self.state)
        story.field_effects.append(impact)
        self._logger.debug(f"Field impact analysis completed for story: {story.id}")

        # Find resonances
        resonances: List[Dict[str, Any]] = (
            await self._resonance_detector.find_resonances(story)
        )
        story.resonances.extend([r["story_id"] for r in resonances])
        self._logger.debug(f"Found {len(resonances)} resonances for story: {story.id}")

        # Store story and update field
        await self._store_story(story)
        await self._update_field_state(story, impact, resonances)
        self._logger.info(f"Story {story.id} added and field state updated")

        return story

    async def _store_story(self, story: Story) -> None:
        """Store story and its embeddings"""
        self._logger.info(f"Storing story: {story.id}")
        embedding = await self._resonance_detector.llm.generate_embedding(story)

        await self._vector_store.store(story, embedding)
        self._logger.info(f"Story {story.id} stored successfully in vector store")
        self._stories[story.id] = story

    async def _update_field_state(
        self, story: Story, impact: Dict, resonances: List[Dict]
    ) -> None:
        """Update field state with enhanced resonance understanding"""

        patterns = await self._analyzer.detect_patterns(
            list(self._stories.values()), self.state
        )

        self._state = FieldState(
            description=impact["analysis"],
            patterns=patterns,
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


# 5. Keep the demo function at the end
async def demo_scenario():
    logger.info("Starting narrative field demonstration...")

    # Initialize components
    llm: LanguageModel = OllamaInterface()
    logger.info(f"Initialized Ollama interface")

    vector_store: VectorStore = ChromaStore(collection_name="research_lab")
    logger.info(f"Initialized Chroma vector store")

    field = NarrativeField(llm, vector_store)
    logger.info(f"Initialized narrative field")

    # Research Lab Scenario with Multiple Characters and 20 Events
    stories = [
        # Event 1: Leon's frustration
        {
            "content": "After enduring a long and tumultuous meeting with his colleagues, where the cacophony of voices made it impossible to think clearly, Leon felt his frustration mounting. The office was so noisy that he couldn't hear himself think, and all he wanted was a moment of peace. Craving solitude, he really wanted to go to lunch without having to wait for the others, hoping that a quiet break would help him regain his composure.",
            "context": "Leon is overwhelmed by the noisy meeting and desires time alone during lunch to clear his mind.",
        },
        # Event 2: Leon discussing the AI minor
        {
            "content": "After lunch, as Leon and Coen walked back to the lab, Leon decided to share his growing concerns about the AI for Society minor. He voiced his doubts and the challenges he foresaw in the program's current direction. Coen listened attentively and was supportive of Leon's worries. \"I think you have some valid points,\" Coen acknowledged, \"but perhaps it would be best to discuss these issues with Danny, the manager of the minor.\" Coen believed that Danny's insights could be crucial in addressing Leon's concerns.",
            "context": "Leon confides in Coen about issues with the AI minor; Coen advises consulting Danny.",
        },
        # Event 3: Danny's accident
        {
            "content": "News spread that Danny had fallen off his bike and was injured. He was on his way to the hospital and unsure if he could continue working on the AI for Society minor in the near future. Leon was deeply worried about Danny's well-being, the impact on the lab, and the future of the AI minor program. Feeling the weight of these concerns, he decided to talk to his manager, Robbert, hoping to find a solution.",
            "context": "Danny's injury raises concerns about the AI minor's future, prompting Leon to seek Robbert's guidance.",
        },
        # Event 4: Robbert's tough advice
        {
            "content": "After work, Robbert and Leon walked back to the lab together. Leon expressed his worries about Danny's accident and the AI minor. However, Robbert seemed more preoccupied with his own research and was not interested in discussing the minor. \"I know you're concerned, but you need to man up and stop whining,\" Robbert said bluntly. His tough advice left Leon feeling isolated and unsupported.",
            "context": "Robbert dismisses Leon's concerns, focusing instead on his own research priorities.",
        },
        # Event 5: Coen's input
        {
            "content": 'Feeling conflicted after his conversation with Robbert, Leon found solace when Coen offered to help with the AI minor. "Maybe we can work on it together while Danny recovers," Coen suggested. Leon appreciated Coen\'s offer, recognizing the value of teamwork, but he also felt uncertain about taking on more responsibility without proper guidance.',
            "context": "Coen volunteers to assist Leon with the AI minor during Danny's absence.",
        },
        # Event 6: Sarah’s contribution
        {
            "content": "Sarah, a new member of the lab eager to make her mark, approached Leon with a fresh idea. Enthusiastic about the ethical challenges in AI, she suggested a new direction for the AI minor—focusing on ethics in AI development. Her excitement was contagious, and Leon began to see the potential impact of integrating ethics into the program.",
            "context": "Sarah proposes refocusing the AI minor on AI ethics, sparking interest from Leon.",
        },
        # Event 7: Tom's exhaustion
        {
            "content": "Tom, another member of the lab, was visibly exhausted after a long day. He had been struggling to keep up with the heavy workload and confided in his colleagues that he wanted to leave early. Considering taking a break from the lab altogether, Tom felt mentally drained and knew he needed time to recover.",
            "context": "Tom is overwhelmed by work stress and thinks about temporarily leaving the lab.",
        },
        # Event 8: Leon reassessing
        {
            "content": "Observing Tom's exhaustion, Leon became concerned that the lab might be overworking its members. Balancing his worries about the AI minor and the well-being of his colleagues, he suggested organizing a team meeting to discuss workload management. Leon hoped that addressing these issues openly would help prevent burnout and improve overall productivity.",
            "context": "Leon considers holding a meeting to tackle workload issues affecting team morale.",
        },
        # Event 9: Robbert's counter
        {
            "content": "Robbert disagreed with Leon's assessment, arguing that the lab members needed to toughen up and handle the workload. He felt that reducing their responsibilities would slow down progress on important research projects. \"We can't afford to ease up now,\" Robbert insisted, dismissing the idea of altering the current work demands.",
            "context": "Robbert rejects the notion of reducing workloads, emphasizing the need for ongoing productivity.",
        },
        # Event 10: Coen's personal struggle
        {
            "content": "In a candid conversation, Coen revealed to Leon that he had been dealing with personal issues and was struggling to focus on work. Leon was surprised by Coen's admission, as he had always appeared to have everything under control. This revelation highlighted the underlying stress affecting the team.",
            "context": "Coen admits personal struggles are hindering his work, surprising Leon.",
        },
        # Event 11: Sarah's proposal
        {
            "content": "Concerned about her colleagues' mental health, Sarah proposed implementing a flexible working schedule to accommodate those feeling burned out. She believed that a healthier work-life balance would benefit both the individuals and the lab's productivity. \"We need to take care of ourselves to do our best work,\" she advocated.",
            "context": "Sarah suggests flexible hours to improve well-being and efficiency in the lab.",
        },
        # Event 12: Tom’s decision
        {
            "content": "Feeling overwhelmed, Tom decided to take a temporary leave from the lab to focus on his mental health. He believed that stepping back was the best decision for now and hoped that his absence would prompt the team to consider the pressures they were all facing.",
            "context": "Tom takes a break to address his mental health, hoping to highlight team stress.",
        },
        # Event 13: Leon's talk with Robbert
        {
            "content": "Using Tom's situation as an example, Leon tried once more to convince Robbert that the team needed more flexibility. \"If we don't address this, we might lose more valuable team members,\" Leon cautioned. However, Robbert remained unconvinced, believing that the team was coddling itself too much and that personal issues should not interfere with work.",
            "context": "Leon urges Robbert to consider flexibility; Robbert remains steadfast against it.",
        },
        # Event 14: Robbert doubling down
        {
            "content": "Robbert held a team meeting to reiterate the importance of maintaining productivity despite personal challenges. He emphasized that their work was critical and that everyone needed to stay focused. Robbert believed that personal problems should not interfere with lab performance and stood firm on his stance.",
            "context": "Robbert emphasizes productivity over personal issues in a team meeting.",
        },
        # Event 15: Sarah's pushback
        {
            "content": "Sarah pushed back against Robbert's position during the meeting, arguing that a more flexible approach would ultimately lead to better results. She highlighted the risks of burnout and the benefits of supporting team members through their personal struggles. The team found itself divided between Robbert's hardline approach and Sarah's call for change.",
            "context": "Sarah challenges Robbert's views, leading to a team split over work policies.",
        },
        # Event 16: Coen's suggestion
        {
            "content": 'Seeking a compromise, Coen suggested organizing a workshop on mental health and productivity. "Maybe we can find strategies to balance personal well-being with our work goals," he proposed. Coen hoped this initiative would bring both sides together and foster a more supportive environment.',
            "context": "Coen proposes a mental health workshop to reconcile differing team perspectives.",
        },
        # Event 17: Leon's reflection
        {
            "content": "Leon reflected on the growing tension within the lab and wondered if they needed an external mediator to help resolve the conflicts. Feeling caught between Robbert's expectations and his colleagues' concerns, he contemplated seeking outside assistance to find a constructive path forward.",
            "context": "Leon considers involving a mediator to address internal lab conflicts.",
        },
        # Event 18: A breakthrough idea
        {
            "content": "During a late-night discussion, Leon and Sarah brainstormed a novel approach to restructure the AI minor. They envisioned incorporating elements of ethics and mental health awareness into the curriculum, aligning the program with current societal needs. Energized by this new direction, Leon believed it could address both the challenges facing the AI minor and the lab's workload issues.",
            "context": "Leon and Sarah create a plan integrating ethics and mental health into the AI minor.",
        },
        # Event 19: Robbert's hesitation
        {
            "content": 'When presented with the proposed changes, Robbert was hesitant to implement them. He feared that altering their focus would slow down the lab\'s progress and detract from their primary research objectives. "This plan seems too idealistic," he cautioned, remaining committed to a results-driven approach.',
            "context": "Robbert doubts the practicality of the new AI minor plan, fearing it may impede progress.",
        },
        # Event 20: Tom’s return
        {
            "content": "After his break, Tom returned to the lab feeling refreshed and ready to contribute again. He appreciated the support from his colleagues and felt more optimistic about balancing his mental health with work. Tom's return brought a renewed sense of hope to the team, signaling the potential for positive change.",
            "context": "Tom's rejuvenated return inspires hope for better balance in the lab.",
        },
    ]

    # Process stories
    logger.info(f"Processing {len(stories)} stories and analyzing field effects...")
    for story in stories:
        try:
            await field.add_story(story["content"], story["context"])

        except Exception as e:
            logger.error(f"Error processing story: {e}", exc_info=True)
            continue

    logger.info("Narrative field demonstration completed")


if __name__ == "__main__":
    asyncio.run(demo_scenario())
