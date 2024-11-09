"""
Social Dynamics Simulation System

This module implements a sophisticated multi-agent social simulation that models complex
interpersonal dynamics, emotional states, and group decision-making processes. The system
uses LLMs to generate realistic social interactions based on personality traits,
emotional states, and social status.

Key Components:
1. Social Agents with distinct personalities and emotional states
2. Group dynamics and emotional contagion
3. Decision-making processes influenced by emotions
4. Natural language interaction generation
5. Social status and hierarchy evolution
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import asyncio
from enum import Enum, auto
import numpy as np
import json


@dataclass
class SocialStatus:
    """Represents an agent's social standing with multiple dimensions"""

    formal_rank: float  # Explicit hierarchical position (0-1)
    influence: float  # Ability to affect others' decisions
    respect: float  # Peer-based status
    expertise: Dict[str, float]  # Domain-specific credibility

    def compute_effective_status(self, context: str) -> float:
        """Combines multiple status dimensions based on context"""
        return (
            0.4 * self.influence
            + 0.3 * self.respect
            + 0.2 * self.formal_rank
            + 0.1 * (self.expertise.get(context, 0))
        )


from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PhaseOutcome:
    """Represents the outcome of a deliberation phase.

    Attributes:
        consensus_level: Float between 0 and 1 indicating level of agreement
        key_contributions: List of most significant contributions
        emotional_progress: Dictionary containing emotional progress metrics
    """

    consensus_level: float
    key_contributions: List[str]
    emotional_progress: Dict[str, float]

    def __str__(self) -> str:
        """String representation of phase outcome."""
        return (
            f"Consensus Level: {self.consensus_level:.2f}\n"
            f"Key Contributions:\n"
            + "\n".join(f"- {cont}" for cont in self.key_contributions[:3])
            + f"\nEmotional Progress:\n"
            + "\n".join(f"- {k}: {v:.2f}" for k, v in self.emotional_progress.items())
        )


class InteractionType(Enum):
    SUPPORT = "support"
    CHALLENGE = "challenge"
    DEFER = "defer"
    LEAD = "lead"
    COLLABORATE = "collaborate"


@dataclass
class PersonalityTraits:
    """
    Defines the core personality dimensions that influence social behavior.

    Each trait is represented as a float between 0 and 1, where:
    - extraversion: tendency to seek and enjoy social interactions
    - agreeableness: tendency to be cooperative and maintain harmony
    - dominance: tendency to assert control and influence others
    - openness: receptiveness to new ideas and experiences
    - stability: emotional consistency and resilience

    These traits influence how agents:
    1. Choose interaction strategies
    2. Respond to emotional situations
    3. Make decisions in group contexts
    4. Build and maintain relationships
    """

    extraversion: float  # 0-1: tendency to seek social interactions
    agreeableness: float  # 0-1: tendency to maintain social harmony
    dominance: float  # 0-1: tendency to assert control
    openness: float  # 0-1: receptiveness to new ideas/situations
    stability: float  # 0-1: emotional consistency

    def to_dict(self) -> Dict[str, float]:
        return {
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "dominance": self.dominance,
            "openness": self.openness,
            "stability": self.stability,
        }


class LLMInterface:
    def __init__(self, model: str):
        self.model = model
        self.conversation_history: List[Dict] = []

    async def generate(self, prompt: str) -> str:
        """Interface with LLM API and return response"""
        # Implementation depends on specific LLM being used
        pass

    def _format_personality(self, personality: PersonalityTraits) -> str:
        """Format personality traits for prompt inclusion"""
        traits = personality.to_dict()
        return ", ".join([f"{k}: {v:.2f}" for k, v in traits.items()])


class EmotionalState:
    """Represents and manages an agent's emotional state"""

    class Emotion(Enum):
        HAPPY = "happy"
        ANGRY = "angry"
        ANXIOUS = "anxious"
        CONFIDENT = "confident"
        DEFENSIVE = "defensive"
        NEUTRAL = "neutral"

    def __init__(self, stability: float):
        self.stability = stability
        self.current_emotions = {emotion: 0.0 for emotion in self.Emotion}
        self.current_emotions[self.Emotion.NEUTRAL] = 0.5  # Start with neutral state

    def update_emotion(self, emotion: Emotion, intensity: float) -> None:
        """Update the intensity of a specific emotion.

        Args:
            emotion: The emotion to update
            intensity: New intensity value (0-1)
        """
        # Ensure intensity is within bounds
        intensity = max(0.0, min(1.0, intensity))

        # Apply stability factor to change
        current = self.current_emotions.get(emotion, 0.0)
        change = intensity - current
        stabilized_change = change * (1 - self.stability)

        # Update emotion
        self.current_emotions[emotion] = current + stabilized_change

        # Normalize other emotions to maintain total intensity
        self._normalize_emotions(emotion)

    def _normalize_emotions(self, primary_emotion: Emotion) -> None:
        """Normalize emotion intensities to maintain reasonable total intensity."""
        total = sum(self.current_emotions.values())
        if total > 1.0:
            # Scale down other emotions proportionally
            scale_factor = (1.0 - self.current_emotions[primary_emotion]) / (
                total - self.current_emotions[primary_emotion]
            )
            for emotion in self.current_emotions:
                if emotion != primary_emotion:
                    self.current_emotions[emotion] *= scale_factor

    def get_dominant_emotion(self) -> Tuple[Emotion, float]:
        """Returns the strongest current emotion and its intensity"""
        dominant = max(self.current_emotions.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]


@dataclass
class InteractionOutcome:
    """Represents the result of a social interaction"""

    success: float  # 0-1: how well the interaction achieved its goals
    status_change: float  # change in status (-1 to 1)
    emotional_impact: Dict[EmotionalState.Emotion, float]
    interaction_text: str


class InteractionMemory:
    """Stores and manages agent's memory of past interactions"""

    def __init__(self):
        self.interactions: List[Dict] = []

    def add_interaction(
        self,
        other_agent_id: str,
        interaction_text: str,
        outcome: "InteractionOutcome",
        emotional_state: Optional[Tuple[EmotionalState.Emotion, float]] = None,
    ) -> None:
        """Add a new interaction to memory"""
        self.interactions.append(
            {
                "agent_id": other_agent_id,
                "text": interaction_text,
                "outcome": outcome,
                "emotional_state": emotional_state,
                "timestamp": len(self.interactions),  # Simple timestamp for ordering
            }
        )

    def get_recent_interactions(self, n: int = 5) -> List[Dict]:
        """Get the n most recent interactions"""
        return self.interactions[-n:]

    def get_interactions_with(self, agent_id: str) -> List[Dict]:
        """Get all interactions with a specific agent"""
        return [
            interaction
            for interaction in self.interactions
            if interaction["agent_id"] == agent_id
        ]


class SocialAgent:
    """
    Represents an individual agent in the social simulation.

    Each agent has:
    1. Unique personality traits
    2. Social status and relationships
    3. Emotional state and memory
    4. Decision-making capabilities

    Agents can:
    1. Generate and execute social interactions
    2. Form and maintain relationships
    3. Participate in group decision-making
    4. Learn from social experiences
    """

    def __init__(
        self,
        agent_id: str,
        llm_model: str,
        initial_status: Optional[SocialStatus] = None,
        personality_traits: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize a social agent with specified characteristics.

        Args:
            agent_id: Unique identifier for the agent
            llm_model: Language model to use for interaction generation
            initial_status: Starting social status (optional)
            personality_traits: Initial personality configuration (optional)
        """
        self.id = agent_id
        self.status = initial_status or SocialStatus(
            formal_rank=0.5, influence=0.5, respect=0.5, expertise={}
        )
        self.relationships: Dict[str, float] = {}  # agent_id -> relationship_quality
        self.memory = InteractionMemory()
        self.llm = LLMInterface(model=llm_model)
        self.personality = personality_traits or self._generate_personality()

    async def decide_interaction(
        self, other_agent: "SocialAgent", context: str
    ) -> InteractionType:
        """Determine interaction strategy using LLM"""
        prompt = PromptTemplate.INTERACTION_DECISION.format(
            context=context,
            agent1_status=self._format_status(),
            agent1_personality=self.llm._format_personality(self.personality),
            agent1_recent_history=self.memory.get_recent_interactions(5),
            agent2_status=other_agent._format_status(),
            relationship_quality=self.relationships.get(other_agent.id, 0.5),
            interaction_history=self.memory.get_interactions_with(other_agent.id),
        )

        response = await self.llm.generate(prompt)
        try:
            decision = json.loads(response)
            return InteractionType(decision["strategy"])
        except (json.JSONDecodeError, KeyError):
            return InteractionType.COLLABORATE  # Safe default

    async def generate_interaction_text(
        self, other_agent: "SocialAgent", strategy: InteractionType, context: str
    ) -> str:
        """Generate natural language interaction based on chosen strategy"""
        prompt = PromptTemplate.INTERACTION_GENERATION.format(
            context=context,
            strategy=strategy.value,
            relationship_details=self._format_relationship_details(other_agent),
            agent_details=self._format_agent_details(),
            target_details=other_agent._format_agent_details(),
        )

        return await self.llm.generate(prompt)

    async def interact(
        self, other_agent: "SocialAgent", context: str
    ) -> InteractionOutcome:
        """Execute a single social interaction between agents"""
        # Determine relative status and power dynamics
        status_diff = self.status.compute_effective_status(
            context
        ) - other_agent.status.compute_effective_status(context)

        # Choose interaction strategy based on status difference and personality
        interaction_type = await self.decide_interaction(other_agent, context)

        # Generate natural language interaction
        interaction_text = await self._generate_interaction_text(
            other_agent, interaction_type, context
        )

        # Update social metrics based on interaction outcome
        outcome = self._process_interaction(
            other_agent, interaction_type, interaction_text, status_diff
        )

        # Update memories and relationships
        self.memory.add_interaction(other_agent.id, interaction_text, outcome)
        self._update_relationship(other_agent.id, outcome)

        return outcome

    def _generate_personality(self) -> PersonalityTraits:
        """Generate default personality traits"""
        return PersonalityTraits(
            extraversion=0.5,  # Neutral default values
            agreeableness=0.5,
            dominance=0.5,
            openness=0.5,
            stability=0.5,
        )

    def _format_status(self) -> str:
        """Format status for prompt inclusion"""
        return f"""
        Formal Rank: {self.status.formal_rank:.2f}
        Influence: {self.status.influence:.2f}
        Respect: {self.status.respect:.2f}
        Expertise: {', '.join(f'{k}: {v:.2f}' for k, v in self.status.expertise.items())}
        """

    def _format_agent_details(self) -> str:
        """Format agent details for prompt inclusion"""
        return f"""
        Agent ID: {self.id}
        Status: {self._format_status()}
        Personality: {self.llm._format_personality(self.personality)}
        Recent History: {self.memory.get_recent_interactions(3)}
        """

    def _format_relationship_details(self, other_agent: "SocialAgent") -> str:
        """Format relationship details for prompt inclusion"""
        return f"""
        Relationship Quality: {self.relationships.get(other_agent.id, 0.5):.2f}
        Interaction History: {self.memory.get_interactions_with(other_agent.id)}
        Status Differential: {self.status.compute_effective_status("general") - 
                            other_agent.status.compute_effective_status("general"):.2f}
        """


class SocialHierarchySimulation:
    def __init__(self, num_agents: int, contexts: List[str], llm_model: str = "gpt-4"):
        self.agents = self._initialize_agents(num_agents, llm_model)
        self.contexts = contexts
        self.interaction_history: List[InteractionOutcome] = []

    async def run_simulation_step(self):
        """Execute one step of the simulation where multiple agents interact"""
        interactions = []
        context = np.random.choice(self.contexts)

        # Select agent pairs for interaction
        interaction_pairs = self._select_interaction_pairs()

        # Process interactions in parallel
        tasks = [
            self._process_agent_interaction(agent1, agent2, context)
            for agent1, agent2 in interaction_pairs
        ]
        interactions = await asyncio.gather(*tasks)

        self.interaction_history.extend(interactions)
        self._update_global_hierarchy()

    def analyze_hierarchy(self) -> Dict:
        """Analyze the current state of the social hierarchy"""
        return {
            "status_distribution": self._compute_status_distribution(),
            "influence_networks": self._map_influence_networks(),
            "power_dynamics": self._analyze_power_dynamics(),
            "emergent_leaders": self._identify_emergent_leaders(),
        }


class Coalition:
    def __init__(self, founding_members: Set[SocialAgent]):
        self.members = founding_members
        self.collective_power = self._compute_coalition_power()
        self.shared_goals = self._identify_shared_goals()


class StatusChallenge:
    def __init__(self, challenger: SocialAgent, target: SocialAgent):
        self.challenger = challenger
        self.target = target
        self.supporters: Dict[SocialAgent, float] = {}  # agent -> support_strength


class InteractionStyle:
    """Generates personality-consistent interaction patterns"""

    def __init__(self, personality: PersonalityTraits):
        self.personality = personality
        self.style_patterns = self._generate_style_patterns()

    def _generate_style_patterns(self) -> Dict[str, float]:
        """Create interaction style weights based on personality"""
        return {
            "directness": self._calculate_directness(),
            "formality": self._calculate_formality(),
            "warmth": self._calculate_warmth(),
            "assertiveness": self._calculate_assertiveness(),
            "detail_orientation": self._calculate_detail_orientation(),
        }

    def _calculate_directness(self) -> float:
        """Higher for dominant, lower for agreeable personalities"""
        return 0.7 * self.personality.dominance + 0.3 * (
            1 - self.personality.agreeableness
        )

    def _calculate_formality(self) -> float:
        """Higher for stable, lower for extraverted personalities"""
        return 0.6 * self.personality.stability + 0.4 * (
            1 - self.personality.extraversion
        )

    def _calculate_warmth(self) -> float:
        """Higher for agreeable and extraverted personalities"""
        return (
            0.6 * self.personality.agreeableness + 0.4 * self.personality.extraversion
        )

    def _calculate_assertiveness(self) -> float:
        """Higher for dominant and extraverted personalities"""
        return 0.7 * self.personality.dominance + 0.3 * self.personality.extraversion

    def _calculate_detail_orientation(self) -> float:
        """Higher for stable and less open personalities"""
        return 0.6 * self.personality.stability + 0.4 * (1 - self.personality.openness)

    def modify_prompt(self, base_prompt: str, emotional_state: EmotionalState) -> str:
        """Adjust interaction prompt based on personality and emotional state"""
        dominant_emotion, intensity = emotional_state.get_dominant_emotion()

        style_guidance = f"""
        Adjust the response to reflect:
        - Directness: {self.style_patterns['directness']:.2f}
        - Formality: {self.style_patterns['formality']:.2f}
        - Warmth: {self.style_patterns['warmth']:.2f}
        - Assertiveness: {self.style_patterns['assertiveness']:.2f}
        - Detail orientation: {self.style_patterns['detail_orientation']:.2f}
        
        Current emotional state: {dominant_emotion.value} (intensity: {intensity:.2f})
        
        Maintain consistency with this personality and emotional profile while generating the response.
        """

        return base_prompt + style_guidance


class EnhancedSocialAgent(SocialAgent):
    """Enhanced version of SocialAgent with emotional state and interaction style"""

    def __init__(
        self,
        agent_id: str,
        llm_model: str,
        personality: Optional[PersonalityTraits] = None,
        initial_status: Optional[SocialStatus] = None,
    ):
        # Initialize with default status if none provided
        default_status = SocialStatus(
            formal_rank=0.5, influence=0.5, respect=0.5, expertise={}
        )

        # Call parent constructor with all necessary parameters
        super().__init__(
            agent_id=agent_id,
            llm_model=llm_model,
            initial_status=initial_status or default_status,
            personality_traits=personality,
        )

        self.emotional_state = EmotionalState(self.personality.stability)
        self.interaction_style = InteractionStyle(self.personality)

    async def generate_interaction_text(
        self, other_agent: "SocialAgent", strategy: InteractionType, context: str
    ) -> str:
        """Generate styled interaction text based on personality and emotional state"""
        base_prompt = PromptTemplate.INTERACTION_GENERATION.format(
            context=context,
            strategy=strategy.value,
            relationship_details=self._format_relationship_details(other_agent),
            agent_details=self._format_agent_details(),
            target_details=other_agent._format_agent_details(),
        )

        styled_prompt = self.interaction_style.modify_prompt(
            base_prompt, self.emotional_state
        )

        return await self.llm.generate(styled_prompt)

    async def process_interaction_outcome(
        self, outcome: "InteractionOutcome", other_agent: "SocialAgent"
    ) -> None:
        """Process interaction results and update emotional state"""
        self.emotional_state.update_emotion(
            outcome.emotional_impact.get(outcome.emotional_impact.items(), 0.0)
        )
        self._update_relationship(other_agent.id, outcome)
        self.memory.add_interaction(
            other_agent.id,
            outcome.interaction_text,
            outcome,
            self.emotional_state.get_dominant_emotion(),
        )


class GroupEmotionalDynamics:
    """
    Manages emotional interactions and dynamics within a group of agents.

    This class handles:
    1. Emotional contagion between agents
    2. Group-level emotional states
    3. Subgroup formation and evolution
    4. Relationship network management
    5. Status hierarchy evolution

    The emotional dynamics are influenced by:
    - Individual personality traits
    - Existing relationships
    - Power dynamics
    - Interaction history
    - Current context
    """

    def __init__(self, agents: List[EnhancedSocialAgent]):
        self.agents = agents
        self.emotional_network = self._initialize_emotional_network()
        self.group_emotion = self._calculate_group_emotion()
        self.subgroups: List[Set[EnhancedSocialAgent]] = []
        self.emotional_history: List[Dict] = []

    def _calculate_group_emotion(self) -> Dict[EmotionalState.Emotion, float]:
        """Calculate the aggregate emotional state of the group"""
        group_emotions = {emotion: 0.0 for emotion in EmotionalState.Emotion}

        # Sum up emotional states weighted by influence
        total_influence = 0.0
        for agent in self.agents:
            influence = agent.status.influence
            total_influence += influence

            # Get agent's emotional state
            for emotion, intensity in agent.emotional_state.current_emotions.items():
                group_emotions[emotion] += intensity * influence

        # Normalize by total influence
        if total_influence > 0:
            for emotion in group_emotions:
                group_emotions[emotion] /= total_influence

        return group_emotions

    def _initialize_emotional_network(self) -> Dict:
        """Initialize emotional influence network between agents"""
        return {
            agent.id: {
                other.id: self._calculate_emotional_influence(agent, other)
                for other in self.agents
                if other != agent
            }
            for agent in self.agents
        }

    def _calculate_emotional_influence(
        self, agent1: EnhancedSocialAgent, agent2: EnhancedSocialAgent
    ) -> float:
        """Calculate emotional influence strength between two agents"""
        return (
            0.4 * agent1.status.influence
            + 0.3 * agent1.personality.extraversion
            + 0.3 * (1 - agent2.personality.stability)
        )

    async def process_group_interaction(
        self, interaction: "GroupInteraction"
    ) -> Dict[str, EmotionalState]:
        """Process emotional effects of a group interaction"""
        # Update emotional states of all involved agents
        emotional_responses = await self._propagate_emotions(interaction)

        # Update group-level emotions
        self.group_emotion = self._calculate_group_emotion()

        # Detect and update emotional subgroups
        self._update_subgroups()

        # Record emotional state
        self._record_emotional_state()

        return emotional_responses  # Make sure to return the responses

    def _record_emotional_state(self) -> None:
        """Record current emotional state in history"""
        self.emotional_history.append(
            {
                "group_emotion": self.group_emotion.copy(),
                "timestamp": len(self.emotional_history),
                "subgroups": [
                    {
                        "members": [agent.id for agent in subgroup],
                        "primary_emotion": self._get_subgroup_primary_emotion(subgroup),
                    }
                    for subgroup in self.subgroups
                ],
            }
        )

    def _get_subgroup_primary_emotion(
        self, subgroup: Set[EnhancedSocialAgent]
    ) -> EmotionalState.Emotion:
        """Calculate the dominant emotion in a subgroup"""
        emotion_sums = {emotion: 0.0 for emotion in EmotionalState.Emotion}

        for agent in subgroup:
            for emotion, intensity in agent.emotional_state.current_emotions.items():
                emotion_sums[emotion] += intensity

        return max(emotion_sums.items(), key=lambda x: x[1])[0]

    def _update_subgroups(self) -> None:
        """Update emotional subgroup formations"""
        # Implementation of subgroup detection and updating
        # This could use clustering based on emotional states and relationships
        pass

    async def _propagate_emotions(
        self, interaction: "GroupInteraction"
    ) -> Dict[str, EmotionalState]:
        """Propagate emotional effects through the group based on interaction"""
        emotional_responses = {}

        # Process initiator's emotional impact
        initiator_emotion = await self._process_initiator_emotion(interaction)
        emotional_responses[interaction.initiator.id] = initiator_emotion

        # Process participants' emotional responses
        for participant in interaction.participants:
            if participant != interaction.initiator:
                response = await self._process_participant_emotion(
                    participant, interaction, initiator_emotion
                )
                emotional_responses[participant.id] = response

        # Calculate emotional contagion effects
        await self._process_emotional_contagion(emotional_responses, interaction)

        return emotional_responses

    async def _process_initiator_emotion(
        self, interaction: "GroupInteraction"
    ) -> EmotionalState:
        """Process the emotional state of the interaction initiator"""
        initiator = interaction.initiator

        # Base emotional response based on interaction type
        base_emotion = self._get_base_emotion_for_type(interaction.type)

        # Modify based on personality and status
        intensity = (
            0.5  # Base intensity
            + 0.3
            * initiator.personality.extraversion  # More expressive personalities have stronger effects
            + 0.2
            * initiator.status.influence  # Higher status increases emotional impact
        )

        # Update initiator's emotional state
        initiator.emotional_state.update_emotion(base_emotion, intensity)

        return initiator.emotional_state

    async def _process_participant_emotion(
        self,
        participant: EnhancedSocialAgent,
        interaction: "GroupInteraction",
        initiator_emotion: EmotionalState,
    ) -> EmotionalState:
        """Process emotional response of a participant"""
        # Calculate influence factor
        influence_factor = self.emotional_network[interaction.initiator.id][
            participant.id
        ]

        # Determine response based on personality and relationship
        response_intensity = (
            influence_factor
            * (
                1 - participant.personality.stability
            )  # Less stable personalities are more affected
            * (
                0.5 + 0.5 * participant.personality.openness
            )  # More open personalities are more responsive
        )

        # Update participant's emotional state
        dominant_emotion, _ = initiator_emotion.get_dominant_emotion()
        participant.emotional_state.update_emotion(dominant_emotion, response_intensity)

        return participant.emotional_state

    async def _process_emotional_contagion(
        self,
        emotional_responses: Dict[str, EmotionalState],
        interaction: "GroupInteraction",
    ) -> None:
        """Process emotional contagion effects through the group"""
        for agent in self.agents:
            if agent.id not in emotional_responses:
                # Calculate contagion effect from all involved agents
                contagion_effect = self._calculate_contagion_effect(
                    agent, emotional_responses, interaction
                )

                # Update observer's emotional state
                for emotion, intensity in contagion_effect.items():
                    current = agent.emotional_state.current_emotions.get(emotion, 0.0)
                    agent.emotional_state.update_emotion(
                        emotion, current + intensity * 0.3
                    )

    def _calculate_contagion_effect(
        self,
        observer: EnhancedSocialAgent,
        emotional_responses: Dict[str, EmotionalState],
        interaction: "GroupInteraction",
    ) -> Dict[EmotionalState.Emotion, float]:
        """Calculate emotional contagion effect on an observer"""
        contagion_effect = {emotion: 0.0 for emotion in EmotionalState.Emotion}
        total_influence = 0.0

        for agent_id, state in emotional_responses.items():
            if agent_id != observer.id:
                influence = self.emotional_network[agent_id][observer.id]
                total_influence += influence

                # Add weighted emotional contribution
                for emotion, intensity in state.current_emotions.items():
                    contagion_effect[emotion] += intensity * influence

        # Normalize effects
        if total_influence > 0:
            for emotion in contagion_effect:
                contagion_effect[emotion] /= total_influence

        return contagion_effect

    def _get_base_emotion_for_type(
        self, interaction_type: "GroupInteraction.Type"
    ) -> EmotionalState.Emotion:
        """Map interaction type to base emotion"""
        emotion_map = {
            GroupInteraction.Type.ANNOUNCEMENT: EmotionalState.Emotion.NEUTRAL,
            GroupInteraction.Type.DISCUSSION: EmotionalState.Emotion.CONFIDENT,
            GroupInteraction.Type.CONFLICT: EmotionalState.Emotion.ANGRY,
            GroupInteraction.Type.CELEBRATION: EmotionalState.Emotion.HAPPY,
            GroupInteraction.Type.CRISIS: EmotionalState.Emotion.ANXIOUS,
        }
        return emotion_map.get(interaction_type, EmotionalState.Emotion.NEUTRAL)

    def analyze_hierarchy(self) -> Dict[str, Any]:
        """Analyze the current social hierarchy and group dynamics.

        Returns:
            Dictionary containing analysis of group state including:
            - cohesion_level: Overall group cohesion
            - relationship_changes: Notable relationship changes
            - emergent_leaders: Agents who have emerged as informal leaders
        """
        # Calculate group cohesion
        cohesion_level = self._calculate_group_cohesion()

        # Analyze relationship changes
        relationship_changes = self._analyze_relationship_changes()

        # Identify emergent leaders
        emergent_leaders = self._identify_emergent_leaders()

        return {
            "cohesion_level": cohesion_level,
            "relationship_changes": relationship_changes,
            "emergent_leaders": emergent_leaders,
        }

    def _calculate_group_cohesion(self) -> float:
        """Calculate overall group cohesion level."""
        if not self.agents:
            return 0.0

        # Calculate average relationship strength
        total_strength = 0.0
        relationship_count = 0

        for agent in self.agents:
            for other in self.agents:
                if agent.id != other.id:
                    strength = agent.relationships.get(other.id, 0.5)
                    total_strength += strength
                    relationship_count += 1

        relationship_cohesion = (
            total_strength / relationship_count if relationship_count > 0 else 0.0
        )

        # Consider emotional alignment
        emotional_alignment = self._calculate_emotional_alignment()

        # Combine factors
        return 0.6 * relationship_cohesion + 0.4 * emotional_alignment

    def _calculate_emotional_alignment(self) -> float:
        """Calculate emotional alignment between group members."""
        if not self.agents:
            return 0.0

        # Get dominant emotions for each agent
        dominant_emotions = {}
        for agent in self.agents:
            emotion, _ = agent.emotional_state.get_dominant_emotion()
            dominant_emotions[agent.id] = emotion

        # Calculate proportion of agents sharing the most common emotion
        if not dominant_emotions:
            return 0.0

        most_common = max(
            set(dominant_emotions.values()), key=list(dominant_emotions.values()).count
        )
        return list(dominant_emotions.values()).count(most_common) / len(
            dominant_emotions
        )

    def _analyze_relationship_changes(self) -> List[str]:
        """Analyze significant changes in relationships."""
        changes = []

        for agent in self.agents:
            for other in self.agents:
                if agent.id != other.id:
                    current = agent.relationships.get(other.id, 0.5)
                    # Note significant relationship changes
                    if current > 0.8:
                        changes.append(
                            f"Strong bond formed between {agent.id} and {other.id}"
                        )
                    elif current < 0.3:
                        changes.append(
                            f"Relationship strain between {agent.id} and {other.id}"
                        )

        return changes

    def _identify_emergent_leaders(self) -> List[str]:
        """Identify agents who have emerged as informal leaders."""
        leaders = []

        for agent in self.agents:
            # Calculate leadership score based on status and relationships
            influence_score = agent.status.influence
            respect_score = agent.status.respect

            # Calculate average relationship strength
            relationship_strength = (
                sum(agent.relationships.values()) / len(agent.relationships)
                if agent.relationships
                else 0
            )

            leadership_score = (
                0.4 * influence_score
                + 0.3 * respect_score
                + 0.3 * relationship_strength
            )

            if leadership_score > 0.7:  # Threshold for identifying leaders
                leaders.append(f"{agent.id} (score: {leadership_score:.2f})")

        return leaders


class GroupInteraction:
    """Represents a group-level interaction"""

    class Type(Enum):
        ANNOUNCEMENT = auto()
        DISCUSSION = auto()
        CONFLICT = auto()
        CELEBRATION = auto()
        CRISIS = auto()

    def __init__(
        self,
        interaction_type: Type,
        initiator: EnhancedSocialAgent,
        participants: Set[EnhancedSocialAgent],
        context: str,
    ):
        self.type = interaction_type
        self.initiator = initiator
        self.participants = participants
        self.context = context
        self.emotional_impacts: Dict[str, Dict[EmotionalState.Emotion, float]] = {}

    async def execute(self) -> None:
        """Execute the group interaction"""
        # Generate interaction content
        content = await self._generate_interaction_content()
        # Process individual responses
        await self._process_participant_responses(content)
        # Calculate emotional impacts
        self._calculate_emotional_impacts()


class EmotionalSubgroup:
    """Represents a subgroup with shared emotional state"""

    def __init__(
        self, members: Set[EnhancedSocialAgent], primary_emotion: EmotionalState.Emotion
    ):
        self.members = members
        self.primary_emotion = primary_emotion
        self.cohesion = self._calculate_cohesion()
        self.influence = self._calculate_group_influence()

    def _calculate_cohesion(self) -> float:
        """Calculate emotional cohesion within the subgroup"""
        emotional_variance = np.var(
            [
                member.emotional_state.current_emotions[self.primary_emotion]
                for member in self.members
            ]
        )
        return 1 / (1 + emotional_variance)


class GroupEmotionalPromptTemplate:
    """Templates for group emotional interactions"""

    GROUP_EMOTIONAL_RESPONSE = """Generate a group interaction response considering emotional dynamics.

Group Context:
- Dominant Group Emotion: {group_emotion}
- Emotional Cohesion: {cohesion_level}
- Subgroup Dynamics: {subgroup_details}

Initiating Agent:
{initiator_details}
- Emotional State: {initiator_emotion}
- Influence Level: {initiator_influence}

Participant Agents:
{participant_details}

Interaction Type: {interaction_type}
Current Context: {context}

Generate a group interaction that:
1. Reflects the initiator's emotional state and influence
2. Considers existing emotional subgroups
3. Accounts for group-level emotional dynamics
4. Maintains individual personality consistency
5. Shows realistic emotional contagion effects

The interaction should demonstrate how emotions spread and evolve within the group while respecting individual personality differences."""

    SUBGROUP_FORMATION = """Model the formation or modification of emotional subgroups.

Current Subgroups:
{current_subgroups}

Recent Emotional Changes:
{emotional_shifts}

Group Tension Points:
{tension_details}

Generate subgroup dynamics that:
1. Show realistic emotional clustering
2. Reflect personality-based grouping tendencies
3. Account for status and influence effects
4. Maintain plausible group sizes
5. Consider existing relationships

Output format:
{
    "new_subgroups": [
        {
            "members": ["agent_ids"],
            "primary_emotion": "emotion",
            "cohesion_level": float,
            "formation_reason": "explanation"
        }
    ],
    "dissolved_subgroups": ["subgroup_ids"],
    "modified_subgroups": {
        "subgroup_id": {
            "added_members": ["agent_ids"],
            "removed_members": ["agent_ids"],
            "emotion_shift": "description"
        }
    }
}"""


class ConflictResolution:
    """Manages emotional conflict resolution between agents"""

    class ResolutionStrategy(Enum):
        COMPROMISE = "compromise"
        MEDIATION = "mediation"
        CONFRONTATION = "confrontation"
        AVOIDANCE = "avoidance"
        ACCOMMODATION = "accommodation"

    def __init__(self, group_dynamics: GroupEmotionalDynamics):
        self.group = group_dynamics
        self.active_conflicts: Dict[str, "Conflict"] = {}
        self.resolution_history: List[Dict] = []

    async def resolve_conflict(self, conflict: "Conflict") -> "ResolutionOutcome":
        """Manage the conflict resolution process"""
        strategy = self._determine_resolution_strategy(conflict)
        resolution_plan = await self._generate_resolution_plan(conflict, strategy)

        return await self._execute_resolution(resolution_plan)

    def _determine_resolution_strategy(
        self, conflict: "Conflict"
    ) -> ResolutionStrategy:
        """Choose appropriate resolution strategy based on participants and context"""
        participants_traits = [agent.personality for agent in conflict.participants]

        # Calculate strategy weights based on participant personalities
        strategy_weights = {
            self.ResolutionStrategy.COMPROMISE: self._calculate_compromise_weight(
                participants_traits
            ),
            self.ResolutionStrategy.MEDIATION: self._calculate_mediation_weight(
                participants_traits
            ),
            self.ResolutionStrategy.CONFRONTATION: self._calculate_confrontation_weight(
                participants_traits
            ),
            self.ResolutionStrategy.AVOIDANCE: self._calculate_avoidance_weight(
                participants_traits
            ),
            self.ResolutionStrategy.ACCOMMODATION: self._calculate_accommodation_weight(
                participants_traits
            ),
        }

        return max(strategy_weights.items(), key=lambda x: x[1])[0]


@dataclass
class Conflict:
    """Represents an emotional conflict between agents"""

    id: str
    participants: Set[EnhancedSocialAgent]
    trigger: str
    emotional_states: Dict[str, EmotionalState]
    intensity: float
    context: str

    def calculate_resolution_difficulty(self) -> float:
        """Calculate how difficult the conflict will be to resolve"""
        return (
            self.intensity
            * self._emotional_divergence()
            * self._personality_incompatibility()
        )


@dataclass
class DecisionProposal:
    """Represents a decision to be made by the group"""

    topic: str
    options: List[str]
    initiator: EnhancedSocialAgent
    context: str
    urgency: float
    emotional_stakes: Optional[Dict[EmotionalState.Emotion, float]] = None

    def __post_init__(self):
        if self.emotional_stakes is None:
            self.emotional_stakes = self._calculate_emotional_stakes()

    def _calculate_emotional_stakes(self) -> Dict[EmotionalState.Emotion, float]:
        """Calculate emotional importance of the decision"""
        return {emotion: 0.0 for emotion in EmotionalState.Emotion}


@dataclass
class DecisionOutcome:
    """Represents the outcome of a group decision-making process.

    Attributes:
        decision: The final decision made
        confidence: Confidence level in the decision (0-1)
        consensus_level: Level of group consensus achieved (0-1)
        emotional_impact: Assessment of emotional effects
        dissenting_agents: List of agents who disagreed with decision
    """

    decision: str
    confidence: float
    consensus_level: float
    emotional_impact: Dict[str, Any]
    dissenting_agents: List[str]

    def __post_init__(self):
        """Validate the outcome attributes."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if not 0 <= self.consensus_level <= 1:
            raise ValueError("Consensus level must be between 0 and 1")
        if not self.decision:
            raise ValueError("Decision cannot be empty")

    def get_implementation_risks(self) -> List[str]:
        """Identify potential risks in implementing the decision."""
        risks = []

        # Check consensus level
        if self.consensus_level < 0.6:
            risks.append("low_consensus")

        # Check emotional impacts
        if self.emotional_impact["resistance_level"] > 0.4:
            risks.append("high_resistance")
        if self.emotional_impact["group_cohesion"] < 0.5:
            risks.append("low_cohesion")

        # Check dissent level
        if (
            len(self.dissenting_agents) > len(self.dissenting_agents) / 3
        ):  # More than 1/3 dissenting
            risks.append("significant_dissent")

        return risks

    def get_support_recommendations(self) -> List[str]:
        """Generate recommendations for supporting decision implementation."""
        recommendations = []

        # Add recommendations based on emotional impact
        if "consensus_building" in self.emotional_impact["support_needs"]:
            recommendations.append("Conduct additional consensus-building sessions")
        if "emotional_support" in self.emotional_impact["support_needs"]:
            recommendations.append("Provide emotional support and reassurance")

        # Add recommendations based on dissent
        if self.dissenting_agents:
            recommendations.append("Address concerns of dissenting members")

        # Add recommendations based on confidence
        if self.confidence < 0.7:
            recommendations.append("Implement additional validation measures")

        return recommendations

    def to_dict(self) -> Dict[str, Any]:
        """Convert the outcome to a dictionary format."""
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "consensus_level": self.consensus_level,
            "emotional_impact": self.emotional_impact,
            "dissenting_agents": self.dissenting_agents,
            "implementation_risks": self.get_implementation_risks(),
            "support_recommendations": self.get_support_recommendations(),
        }


class EmotionalDecisionMaking:
    """Handles group decision-making influenced by emotional states"""

    def __init__(self, group: GroupEmotionalDynamics):
        self.group = group
        self.decision_history: List[Dict] = []
        self.current_proposals: Dict[str, DecisionProposal] = {}

    async def _gather_emotional_responses(
        self, proposal: DecisionProposal
    ) -> Dict[str, Dict]:
        """Gather initial emotional responses to the proposal from all agents.

        Args:
            proposal: The decision proposal to evaluate

        Returns:
            Dictionary mapping agent IDs to their emotional responses
        """
        responses = {}
        for agent in self.group.agents:
            response = await self._get_agent_response(agent, proposal)
            responses[agent.id] = response
        return responses

    async def _get_agent_response(
        self, agent: EnhancedSocialAgent, proposal: DecisionProposal
    ) -> Dict:
        """Get an individual agent's emotional response to a proposal.

        Args:
            agent: The agent providing the response
            proposal: The decision proposal

        Returns:
            Dictionary containing emotional response data
        """
        # Calculate initial reaction based on personality and status
        emotional_reaction = self._calculate_initial_reaction(agent, proposal)

        # Generate specific response to options
        option_responses = await self._evaluate_options(agent, proposal.options)

        return {
            "emotional_reaction": emotional_reaction,
            "option_preferences": option_responses,
            "initial_stance": self._determine_initial_stance(
                emotional_reaction, option_responses
            ),
        }

    def _calculate_initial_reaction(
        self, agent: EnhancedSocialAgent, proposal: DecisionProposal
    ) -> Dict[EmotionalState.Emotion, float]:
        """Calculate agent's initial emotional reaction to proposal."""
        # Initialize with neutral emotion as default
        reaction = {EmotionalState.Emotion.NEUTRAL: 0.5}  # Default neutral state

        # Base reaction on personality traits
        if agent.personality.openness > 0.7:
            reaction[EmotionalState.Emotion.CONFIDENT] = 0.7
        elif agent.personality.stability < 0.4:
            reaction[EmotionalState.Emotion.ANXIOUS] = 0.6

        # Modify based on relationship with initiator
        initiator_relationship = agent.relationships.get(proposal.initiator.id, 0.5)
        if initiator_relationship > 0.7:
            reaction[EmotionalState.Emotion.HAPPY] = 0.5
        elif initiator_relationship < 0.3:
            reaction[EmotionalState.Emotion.DEFENSIVE] = 0.4

        return reaction

    async def _evaluate_options(
        self, agent: EnhancedSocialAgent, options: List[str]
    ) -> Dict[str, float]:
        """Evaluate each option from the agent's perspective.

        Args:
            agent: The agent evaluating options
            options: List of decision options

        Returns:
            Dictionary mapping options to preference scores
        """
        preferences = {}
        for option in options:
            # Calculate preference based on agent's expertise and personality
            preference = self._calculate_option_preference(agent, option)
            preferences[option] = preference
        return preferences

    def _calculate_option_preference(
        self, agent: EnhancedSocialAgent, option: str
    ) -> float:
        """Calculate agent's preference for a specific option.

        Args:
            agent: The agent evaluating
            option: The option being evaluated

        Returns:
            Float between 0 and 1 indicating preference
        """
        # Base preference on relevant expertise
        expertise_factor = max(
            agent.status.expertise.get(domain, 0.0)
            for domain in ["leadership", "technical", "communication"]
        )

        # Modify based on personality traits
        personality_factor = (
            0.4 * agent.personality.openness  # More open personalities prefer change
            + 0.3
            * agent.personality.stability  # Stable personalities prefer structured options
            + 0.3
            * agent.personality.dominance  # Dominant personalities prefer bold options
        )

        return 0.6 * expertise_factor + 0.4 * personality_factor

    def _determine_initial_stance(
        self,
        emotional_reaction: Dict[EmotionalState.Emotion, float],
        option_preferences: Dict[str, float],
    ) -> str:
        """Determine agent's initial stance based on emotions and preferences.

        Args:
            emotional_reaction: Dictionary of emotional reactions
            option_preferences: Dictionary of option preferences

        Returns:
            String indicating initial stance
        """
        # Calculate average option preference
        avg_preference = sum(option_preferences.values()) / len(option_preferences)

        # Get dominant emotion
        dominant_emotion = max(emotional_reaction.items(), key=lambda x: x[1])[0]

        # Determine stance based on emotion and preferences
        if avg_preference > 0.7 and dominant_emotion in [
            EmotionalState.Emotion.HAPPY,
            EmotionalState.Emotion.CONFIDENT,
        ]:
            return "supportive"
        elif avg_preference < 0.3 or dominant_emotion in [
            EmotionalState.Emotion.ANGRY,
            EmotionalState.Emotion.DEFENSIVE,
        ]:
            return "resistant"
        else:
            return "neutral"

    async def make_group_decision(
        self, proposal: "DecisionProposal"
    ) -> "DecisionOutcome":
        """Facilitate emotion-influenced group decision-making"""
        # Gather initial emotional reactions
        emotional_responses = await self._gather_emotional_responses(proposal)

        # Generate decision options considering emotions
        options = await self._generate_emotion_aware_options(
            proposal, emotional_responses
        )

        # Facilitate emotional deliberation
        deliberation_result = await self._emotional_deliberation(
            options, emotional_responses
        )

        # Make final decision
        decision = self._make_final_decision(deliberation_result)

        # Process emotional aftermath
        await self._process_decision_aftermath(decision)

        return decision

    async def _generate_emotion_aware_options(
        self, proposal: DecisionProposal, emotional_responses: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """Generate and evaluate options considering emotional states.

        Args:
            proposal: The current decision proposal
            emotional_responses: Initial emotional responses from agents

        Returns:
            List of option dictionaries with emotional impact assessments
        """
        enhanced_options = []
        for option in proposal.options:
            # Evaluate emotional impact for each agent
            emotional_impacts = {}
            for agent in self.group.agents:
                impact = self._evaluate_emotional_impact(
                    option, agent, emotional_responses[agent.id]
                )
                emotional_impacts[agent.id] = impact

            enhanced_options.append(
                {
                    "option": option,
                    "emotional_impacts": emotional_impacts,
                    "group_alignment": self._calculate_group_alignment(
                        emotional_impacts
                    ),
                    "risk_level": self._assess_emotional_risk(emotional_impacts),
                }
            )

        return enhanced_options

    def _evaluate_emotional_impact(
        self, option: str, agent: EnhancedSocialAgent, current_response: Dict
    ) -> Dict[str, float]:
        """Evaluate potential emotional impact of an option on an agent."""
        # Consider agent's personality and current emotional state
        stability_factor = agent.personality.stability
        openness_factor = agent.personality.openness

        return {
            "acceptance": 0.5
            + (0.3 * openness_factor),  # Base acceptance modified by openness
            "stress": 0.3
            * (1 - stability_factor),  # Lower stability means higher stress
            "enthusiasm": 0.4
            * agent.personality.extraversion,  # Extraverts show more enthusiasm
        }

    def _calculate_group_alignment(
        self, emotional_impacts: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate how well an option aligns with group emotions."""
        if not emotional_impacts:
            return 0.0

        # Average acceptance across all agents
        acceptance_scores = [
            impacts["acceptance"] for impacts in emotional_impacts.values()
        ]
        return sum(acceptance_scores) / len(acceptance_scores)

    def _assess_emotional_risk(
        self, emotional_impacts: Dict[str, Dict[str, float]]
    ) -> float:
        """Assess emotional risk level of an option."""
        if not emotional_impacts:
            return 0.0

        # Consider stress levels and low acceptance as risks
        risk_factors = []
        for impacts in emotional_impacts.values():
            risk = impacts["stress"] * (1 - impacts["acceptance"])
            risk_factors.append(risk)

        return sum(risk_factors) / len(risk_factors)

    async def _emotional_deliberation(
        self, options: List[Dict[str, Any]], emotional_responses: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Facilitate emotional deliberation process for decision options.

        Args:
            options: List of emotion-aware options
            emotional_responses: Initial emotional responses from agents

        Returns:
            Dictionary containing deliberation results
        """
        # Track deliberation state
        deliberation_state = {
            "rounds": [],
            "current_preferences": {},
            "emotional_shifts": {},
            "consensus_level": 0.0,
        }

        # Run deliberation rounds until consensus or max rounds reached
        max_rounds = 3
        for round_num in range(max_rounds):
            # Get contributions from all agents
            round_contributions = await self._gather_round_contributions(
                options, deliberation_state
            )

            # Process emotional dynamics
            emotional_shifts = await self._process_round_dynamics(
                round_contributions, deliberation_state
            )

            # Update preferences based on contributions
            new_preferences = self._update_preferences(
                round_contributions, emotional_shifts
            )

            # Calculate consensus level
            consensus_level = self._calculate_round_consensus(new_preferences)

            # Update deliberation state
            deliberation_state["rounds"].append(
                {
                    "contributions": round_contributions,
                    "emotional_shifts": emotional_shifts,
                    "preferences": new_preferences,
                    "consensus_level": consensus_level,
                }
            )
            deliberation_state["current_preferences"] = new_preferences
            deliberation_state["emotional_shifts"].update(emotional_shifts)
            deliberation_state["consensus_level"] = consensus_level

            # Check if sufficient consensus reached
            if consensus_level > 0.8:  # 80% consensus threshold
                break

        return deliberation_state

    async def _gather_round_contributions(
        self, options: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Gather contributions from all agents for current round."""
        contributions = []
        for agent in self.group.agents:
            contribution = await self._get_agent_contribution(agent, options, state)
            contributions.append(contribution)
        return contributions

    async def _get_agent_contribution(
        self,
        agent: EnhancedSocialAgent,
        options: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get single agent's contribution for current round."""
        # Consider agent's personality and emotional state
        current_emotion = agent.emotional_state.get_dominant_emotion()

        return {
            "agent_id": agent.id,
            "preferred_option": self._determine_agent_preference(agent, options),
            "emotional_state": current_emotion,
            "influence_attempt": self._determine_influence_attempt(agent, state),
        }

    async def _process_round_dynamics(
        self, contributions: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> Dict[str, Dict[EmotionalState.Emotion, float]]:
        """Process emotional dynamics for the current round."""
        emotional_shifts = {}

        # Process each agent's emotional response
        for contribution in contributions:
            agent_id = contribution["agent_id"]
            agent = next(a for a in self.group.agents if a.id == agent_id)

            # Calculate emotional shift based on group dynamics
            shift = self._calculate_emotional_shift(agent, contributions, state)
            emotional_shifts[agent_id] = shift

        return emotional_shifts

    def _update_preferences(
        self,
        contributions: List[Dict[str, Any]],
        emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]],
    ) -> Dict[str, str]:
        """Update agent preferences based on round results."""
        new_preferences = {}

        for contribution in contributions:
            agent_id = contribution["agent_id"]
            # Consider emotional shifts in preference updates
            new_preferences[agent_id] = self._adjust_preference(
                contribution["preferred_option"], emotional_shifts[agent_id]
            )

        return new_preferences

    def _calculate_round_consensus(self, preferences: Dict[str, str]) -> float:
        """Calculate consensus level for current round."""
        if not preferences:
            return 0.0

        # Count option preferences
        option_counts = {}
        for preference in preferences.values():
            option_counts[preference] = option_counts.get(preference, 0) + 1

        # Calculate consensus as proportion supporting most popular option
        max_support = max(option_counts.values())
        return max_support / len(preferences)

    def _determine_agent_preference(
        self, agent: EnhancedSocialAgent, options: List[Dict[str, Any]]
    ) -> str:
        """Determine agent's current option preference."""
        # Score each option based on emotional impact and agent characteristics
        option_scores = {}
        for option in options:
            score = (
                0.4 * option["emotional_impacts"][agent.id]["acceptance"]
                + 0.3 * (1 - option["risk_level"])
                + 0.3 * option["group_alignment"]
            )
            option_scores[option["option"]] = score

        # Return option with highest score
        return max(option_scores.items(), key=lambda x: x[1])[0]

    def _determine_influence_attempt(
        self, agent: EnhancedSocialAgent, state: Dict[str, Any]
    ) -> str:
        """Determine how agent attempts to influence others."""
        if agent.personality.dominance > 0.7:
            return "assertive"
        elif agent.personality.agreeableness > 0.7:
            return "collaborative"
        else:
            return "neutral"

    def _calculate_emotional_shift(
        self,
        agent: EnhancedSocialAgent,
        contributions: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Dict[EmotionalState.Emotion, float]:
        """Calculate emotional shift for an agent based on round dynamics."""
        shifts = {emotion: 0.0 for emotion in EmotionalState.Emotion}

        # Consider influence from other agents
        for contribution in contributions:
            if contribution["agent_id"] != agent.id:
                other_agent = next(
                    a for a in self.group.agents if a.id == contribution["agent_id"]
                )
                influence = self._calculate_emotional_influence(other_agent, agent)

                # Apply emotional influence
                emotion, intensity = contribution["emotional_state"]
                shifts[emotion] += influence * intensity

        return shifts

    def _adjust_preference(
        self,
        current_preference: str,
        emotional_shift: Dict[EmotionalState.Emotion, float],
    ) -> str:
        """Adjust preference based on emotional shifts."""
        # For now, maintain current preference
        # Could be enhanced to allow preference changes based on emotional shifts
        return current_preference

    def _calculate_emotional_influence(
        self, agent1: EnhancedSocialAgent, agent2: EnhancedSocialAgent
    ) -> float:
        """Calculate emotional influence between two agents.

        Args:
            agent1: Source agent (influencer)
            agent2: Target agent (influenced)

        Returns:
            Float between 0 and 1 indicating influence strength
        """
        # Base influence on status and personality
        status_factor = (
            0.4 * agent1.status.influence
            + 0.3 * agent1.status.respect
            + 0.3 * agent1.status.formal_rank
        )

        personality_factor = (
            0.4
            * agent1.personality.extraversion  # More extraverted agents are more influential
            + 0.3
            * agent1.personality.dominance  # More dominant agents have stronger influence
            + 0.3
            * (
                1 - agent2.personality.stability
            )  # Less stable agents are more influenced
        )

        # Consider relationship quality
        relationship_quality = agent1.relationships.get(agent2.id, 0.5)

        # Combine factors with weights
        return (
            0.4 * status_factor + 0.3 * personality_factor + 0.3 * relationship_quality
        )

    def _make_final_decision(
        self, deliberation_result: Dict[str, Any]
    ) -> "DecisionOutcome":
        """Make final decision based on deliberation results.

        Args:
            deliberation_result: Results from the emotional deliberation process

        Returns:
            DecisionOutcome containing the final decision and its implications
        """
        # Get final preferences and emotional states
        final_preferences = deliberation_result["current_preferences"]
        emotional_shifts = deliberation_result["emotional_shifts"]
        consensus_level = deliberation_result["consensus_level"]

        # Determine winning option
        winning_option = self._determine_winning_option(final_preferences)

        # Calculate decision confidence
        confidence = self._calculate_decision_confidence(
            winning_option, final_preferences, emotional_shifts, consensus_level
        )

        # Assess emotional impact
        emotional_impact = self._assess_decision_impact(
            winning_option, emotional_shifts, consensus_level
        )

        # Create decision outcome
        return DecisionOutcome(
            decision=winning_option,
            confidence=confidence,
            consensus_level=consensus_level,
            emotional_impact=emotional_impact,
            dissenting_agents=self._identify_dissenters(
                winning_option, final_preferences
            ),
        )

    def _determine_winning_option(self, preferences: Dict[str, str]) -> str:
        """Determine the winning option based on final preferences."""
        if not preferences:
            raise ValueError("No preferences available for decision")

        # Count votes for each option
        vote_counts = {}
        for preference in preferences.values():
            vote_counts[preference] = vote_counts.get(preference, 0) + 1

        # Return option with most votes
        return max(vote_counts.items(), key=lambda x: x[1])[0]

    def _calculate_decision_confidence(
        self,
        winning_option: str,
        preferences: Dict[str, str],
        emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]],
        consensus_level: float,
    ) -> float:
        """Calculate confidence level in the decision."""
        # Base confidence on consensus level
        base_confidence = consensus_level

        # Adjust based on emotional stability
        emotional_stability = self._calculate_emotional_stability(emotional_shifts)

        # Combine factors
        return 0.7 * base_confidence + 0.3 * emotional_stability

    def _calculate_emotional_stability(
        self, emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]]
    ) -> float:
        """Calculate emotional stability of the group."""
        if not emotional_shifts:
            return 0.0

        # Calculate average magnitude of emotional shifts
        total_shift = 0.0
        count = 0
        for shifts in emotional_shifts.values():
            for shift in shifts.values():
                total_shift += abs(shift)
                count += 1

        avg_shift = total_shift / count if count > 0 else 0.0

        # Convert to stability score (lower shifts = higher stability)
        return 1.0 - min(avg_shift, 1.0)

    def _assess_decision_impact(
        self,
        winning_option: str,
        emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]],
        consensus_level: float,
    ) -> Dict[str, Any]:
        """Assess the emotional impact of the decision."""
        return {
            "group_cohesion": self._calculate_group_cohesion(
                emotional_shifts, consensus_level
            ),
            "emotional_state": self._calculate_group_emotional_state(emotional_shifts),
            "resistance_level": 1.0 - consensus_level,
            "support_needs": self._identify_support_needs(
                emotional_shifts, consensus_level
            ),
        }

    def _calculate_group_cohesion(
        self,
        emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]],
        consensus_level: float,
    ) -> float:
        """Calculate group cohesion level after decision."""
        # Base cohesion on consensus
        base_cohesion = consensus_level

        # Adjust based on emotional alignment
        emotional_alignment = self._calculate_emotional_alignment(emotional_shifts)

        return 0.6 * base_cohesion + 0.4 * emotional_alignment

    def _calculate_group_emotional_state(
        self, emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]]
    ) -> Dict[EmotionalState.Emotion, float]:
        """Calculate final group emotional state."""
        group_state = {emotion: 0.0 for emotion in EmotionalState.Emotion}

        if not emotional_shifts:
            return group_state

        # Aggregate final emotional states
        for shifts in emotional_shifts.values():
            for emotion, shift in shifts.items():
                group_state[emotion] += shift

        # Normalize
        total = sum(abs(v) for v in group_state.values())
        if total > 0:
            for emotion in group_state:
                group_state[emotion] /= total

        return group_state

    def _identify_support_needs(
        self,
        emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]],
        consensus_level: float,
    ) -> List[str]:
        """Identify needed support actions based on decision impact."""
        support_needs = []

        # Check for low consensus
        if consensus_level < 0.6:
            support_needs.append("consensus_building")

        # Check for negative emotional shifts
        negative_emotions = {
            EmotionalState.Emotion.ANGRY,
            EmotionalState.Emotion.ANXIOUS,
            EmotionalState.Emotion.DEFENSIVE,
        }

        for shifts in emotional_shifts.values():
            for emotion, shift in shifts.items():
                if emotion in negative_emotions and shift > 0.3:
                    support_needs.append("emotional_support")
                    break

        return list(set(support_needs))  # Remove duplicates

    def _identify_dissenters(
        self, winning_option: str, preferences: Dict[str, str]
    ) -> List[str]:
        """Identify agents who disagreed with the final decision."""
        return [
            agent_id
            for agent_id, preference in preferences.items()
            if preference != winning_option
        ]

    def _calculate_emotional_alignment(
        self, emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]]
    ) -> float:
        """Calculate emotional alignment between group members.

        Args:
            emotional_shifts: Dictionary of emotional shifts for each agent

        Returns:
            Float between 0 and 1 indicating emotional alignment level
        """
        if not emotional_shifts:
            return 0.0

        # Get dominant emotions for each agent
        dominant_emotions = {}
        for agent_id, shifts in emotional_shifts.items():
            # Find emotion with highest shift
            dominant_emotion = max(shifts.items(), key=lambda x: abs(x[1]))[0]
            dominant_emotions[agent_id] = dominant_emotion

        # Calculate alignment score
        alignment_score = 0.0
        num_pairs = 0

        # Compare each pair of agents
        agents = list(dominant_emotions.keys())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1_id = agents[i]
                agent2_id = agents[j]

                # Check if emotions align
                if dominant_emotions[agent1_id] == dominant_emotions[agent2_id]:
                    # Add full alignment score
                    alignment_score += 1.0
                elif self._are_emotions_compatible(
                    dominant_emotions[agent1_id], dominant_emotions[agent2_id]
                ):
                    # Add partial alignment score for compatible emotions
                    alignment_score += 0.5

                num_pairs += 1

        # Return normalized alignment score
        return alignment_score / num_pairs if num_pairs > 0 else 0.0

    def _are_emotions_compatible(
        self, emotion1: EmotionalState.Emotion, emotion2: EmotionalState.Emotion
    ) -> bool:
        """Determine if two emotions are compatible with each other.

        Args:
            emotion1: First emotion
            emotion2: Second emotion

        Returns:
            Boolean indicating if emotions are compatible
        """
        # Define compatible emotion pairs
        compatible_pairs = {
            (EmotionalState.Emotion.HAPPY, EmotionalState.Emotion.CONFIDENT),
            (EmotionalState.Emotion.ANXIOUS, EmotionalState.Emotion.DEFENSIVE),
            (EmotionalState.Emotion.NEUTRAL, EmotionalState.Emotion.HAPPY),
            (EmotionalState.Emotion.NEUTRAL, EmotionalState.Emotion.CONFIDENT),
        }

        # Check both orderings of the emotion pair
        return (emotion1, emotion2) in compatible_pairs or (
            emotion2,
            emotion1,
        ) in compatible_pairs

    async def _process_decision_aftermath(self, decision: DecisionOutcome) -> None:
        """Process the emotional aftermath of a group decision.

        Args:
            decision: The outcome of the group decision
        """
        # Update emotional states based on decision
        await self._update_emotional_states(decision)

        # Process relationship changes
        self._process_relationship_changes(decision)

        # Record learning outcomes
        self._record_decision_learning(decision)

    async def _update_emotional_states(self, decision: DecisionOutcome) -> None:
        """Update agent emotional states based on decision outcome."""
        for agent in self.group.agents:
            # Check if agent was dissenting
            is_dissenting = agent.id in decision.dissenting_agents

            # Calculate emotional impact
            if is_dissenting:
                # Negative emotional impact for dissenters
                agent.emotional_state.update_emotion(
                    EmotionalState.Emotion.DEFENSIVE, intensity=0.7
                )
            else:
                # Positive emotional impact for supporters
                agent.emotional_state.update_emotion(
                    EmotionalState.Emotion.HAPPY, intensity=0.6
                )

    def _process_relationship_changes(self, decision: DecisionOutcome) -> None:
        """Process relationship changes based on decision outcome."""
        for agent in self.group.agents:
            for other in self.group.agents:
                if agent != other:
                    # Adjust relationships based on agreement/disagreement
                    both_dissenting = (
                        agent.id in decision.dissenting_agents
                        and other.id in decision.dissenting_agents
                    )
                    both_supporting = (
                        agent.id not in decision.dissenting_agents
                        and other.id not in decision.dissenting_agents
                    )

                    if both_dissenting or both_supporting:
                        # Strengthen relationship for aligned agents
                        current = agent.relationships.get(other.id, 0.5)
                        agent.relationships[other.id] = min(1.0, current + 0.1)
                    else:
                        # Weaken relationship for opposed agents
                        current = agent.relationships.get(other.id, 0.5)
                        agent.relationships[other.id] = max(0.0, current - 0.1)

    def _record_decision_learning(self, decision: DecisionOutcome) -> None:
        """Record learning outcomes from the decision process."""
        # Record decision outcome and impact for future reference
        self.decision_history.append(
            {
                "decision": decision.decision,
                "confidence": decision.confidence,
                "consensus_level": decision.consensus_level,
                "emotional_impact": decision.emotional_impact,
                "timestamp": len(self.decision_history),
            }
        )


class EmotionalAftermath:
    """Handles post-decision/conflict emotional ripple effects"""

    class AftermathType(Enum):
        """Types of emotional aftermath effects"""

        RESENTMENT = "resentment"
        STRENGTHENED_BONDS = "strengthened_bonds"
        TRUST_SHIFT = "trust_shift"
        POWER_REBALANCE = "power_rebalance"
        EMOTIONAL_LEARNING = "emotional_learning"

    def __init__(self, group_dynamics: GroupEmotionalDynamics):
        self.group = group_dynamics
        self.active_effects: Dict[str, List["AftermathEffect"]] = {}
        self.learning_history: Dict[str, List["EmotionalLearning"]] = {}

    def _identify_affected_relationships(
        self, event: Union["DecisionOutcome", "ResolutionOutcome"]
    ) -> List[Tuple[str, str]]:
        """Identify relationships affected by the event.

        Args:
            event: The event to analyze

        Returns:
            List of tuples containing agent ID pairs representing affected relationships
        """
        affected_pairs = []

        if isinstance(event, DecisionOutcome):
            # Identify relationships between supporters and dissenters
            supporting_agents = set(
                a.id for a in self.group.agents if a.id not in event.dissenting_agents
            )
            dissenting_agents = set(event.dissenting_agents)

            # Add pairs between supporting and dissenting agents
            for supporter in supporting_agents:
                for dissenter in dissenting_agents:
                    affected_pairs.append((supporter, dissenter))

            # Add pairs between aligned agents (both supporting or both dissenting)
            for group in [supporting_agents, dissenting_agents]:
                for a1 in group:
                    for a2 in group:
                        if a1 < a2:  # Avoid duplicates
                            affected_pairs.append((a1, a2))

        return affected_pairs

    async def _generate_aftermath_effects(
        self,
        event: Union["DecisionOutcome", "ResolutionOutcome"],
        affected_pairs: List[Tuple[str, str]],
    ) -> List["AftermathEffect"]:
        """Generate aftermath effects for affected relationships.

        Args:
            event: The triggering event
            affected_pairs: List of affected relationship pairs

        Returns:
            List of aftermath effects
        """
        effects = []

        for agent1_id, agent2_id in affected_pairs:
            agent1 = next(a for a in self.group.agents if a.id == agent1_id)
            agent2 = next(a for a in self.group.agents if a.id == agent2_id)

            # Determine effect type based on alignment
            if isinstance(event, DecisionOutcome):
                both_dissenting = (
                    agent1_id in event.dissenting_agents
                    and agent2_id in event.dissenting_agents
                )
                both_supporting = (
                    agent1_id not in event.dissenting_agents
                    and agent2_id not in event.dissenting_agents
                )

                if both_dissenting or both_supporting:
                    effect_type = self.AftermathType.STRENGTHENED_BONDS
                else:
                    effect_type = self.AftermathType.TRUST_SHIFT

            effects.append(
                self._create_aftermath_effect(effect_type, agent1, agent2, event)
            )

        return effects

    def _create_aftermath_effect(
        self,
        effect_type: AftermathType,
        agent1: EnhancedSocialAgent,
        agent2: EnhancedSocialAgent,
        event: Union["DecisionOutcome", "ResolutionOutcome"],
    ) -> "AftermathEffect":
        """Create a specific aftermath effect between two agents."""
        intensity = self._calculate_effect_intensity(effect_type, agent1, agent2)
        duration = self._calculate_effect_duration(effect_type, intensity)

        return AftermathEffect(
            effect_type=effect_type,
            agents=(agent1.id, agent2.id),
            intensity=intensity,
            duration=duration,
        )

    def _calculate_effect_intensity(
        self,
        effect_type: AftermathType,
        agent1: EnhancedSocialAgent,
        agent2: EnhancedSocialAgent,
    ) -> float:
        """Calculate the intensity of an aftermath effect."""
        base_intensity = 0.5

        # Adjust based on personality compatibility
        personality_factor = self._calculate_personality_compatibility(
            agent1.personality, agent2.personality
        )

        # Adjust based on current relationship
        relationship_factor = agent1.relationships.get(agent2.id, 0.5)

        return min(
            1.0, base_intensity * (0.7 * personality_factor + 0.3 * relationship_factor)
        )

    def _calculate_effect_duration(
        self, effect_type: AftermathType, intensity: float
    ) -> int:
        """Calculate how long an effect should last (in interaction rounds)."""
        base_duration = {
            self.AftermathType.RESENTMENT: 5,
            self.AftermathType.STRENGTHENED_BONDS: 3,
            self.AftermathType.TRUST_SHIFT: 4,
            self.AftermathType.POWER_REBALANCE: 6,
            self.AftermathType.EMOTIONAL_LEARNING: 8,
        }

        return int(base_duration[effect_type] * (0.5 + 0.5 * intensity))

    def _calculate_personality_compatibility(
        self, personality1: PersonalityTraits, personality2: PersonalityTraits
    ) -> float:
        """Calculate compatibility between two personalities."""
        trait_diffs = [
            abs(personality1.extraversion - personality2.extraversion),
            abs(personality1.agreeableness - personality2.agreeableness),
            abs(personality1.openness - personality2.openness),
            abs(personality1.stability - personality2.stability),
        ]

        avg_diff = sum(trait_diffs) / len(trait_diffs)
        return 1.0 - avg_diff  # Higher compatibility for similar personalities

    async def _apply_emotional_learning(self, effects: List["AftermathEffect"]) -> None:
        """Apply emotional learning from aftermath effects."""
        for effect in effects:
            for agent_id in effect.agents:
                agent = next(a for a in self.group.agents if a.id == agent_id)
                if agent_id not in self.learning_history:
                    self.learning_history[agent_id] = []

                learning = EmotionalLearning(agent)
                await learning.process_experience(effect)
                self.learning_history[agent_id].append(learning)

    def _schedule_long_term_effects(self, effects: List["AftermathEffect"]) -> None:
        """Schedule long-term aftermath effects."""
        for effect in effects:
            for agent_id in effect.agents:
                if agent_id not in self.active_effects:
                    self.active_effects[agent_id] = []
                self.active_effects[agent_id].append(effect)

    async def process_aftermath(
        self, event: Union["DecisionOutcome", "ResolutionOutcome"]
    ) -> None:
        """Process the emotional aftermath of a significant event.

        Args:
            event: The event to process aftermath for
        """
        # Identify affected relationships
        affected_pairs = self._identify_affected_relationships(event)

        # Generate aftermath effects
        effects = await self._generate_aftermath_effects(event, affected_pairs)

        # Apply emotional learning
        await self._apply_emotional_learning(effects)

        # Schedule long-term effects
        self._schedule_long_term_effects(effects)


@dataclass
class AftermathEffect:
    """Represents a specific emotional aftermath effect."""

    effect_type: EmotionalAftermath.AftermathType
    agents: Tuple[str, str]
    intensity: float
    duration: int


class AdvancedDeliberation:
    """
    Manages sophisticated group decision-making processes.

    The deliberation process includes:
    1. Multiple phases of discussion
    2. Emotional state consideration
    3. Power dynamic management
    4. Consensus building
    5. Conflict resolution

    Each phase is designed to:
    - Allow all voices to be heard
    - Consider emotional impacts
    - Build toward consensus
    - Maintain group cohesion
    """

    class DeliberationPhase(Enum):
        """
        Defines the phases of group deliberation.

        EMOTIONAL_SHARING: Initial expression of feelings and concerns
        PERSPECTIVE_BUILDING: Understanding different viewpoints
        SOLUTION_EXPLORATION: Generating and discussing options
        CONSENSUS_BUILDING: Working toward agreement
        COMMITMENT_SECURING: Ensuring buy-in and support
        """

        EMOTIONAL_SHARING = "emotional_sharing"
        PERSPECTIVE_BUILDING = "perspective_building"
        SOLUTION_EXPLORATION = "solution_exploration"
        CONSENSUS_BUILDING = "consensus_building"
        COMMITMENT_SECURING = "commitment_securing"

    def __init__(self, group: GroupEmotionalDynamics):
        self.group = group
        self.current_phase = None
        self.phase_history: List[Dict] = []
        self.emotional_insights: Dict[str, List[str]] = {}

    async def facilitate_phase(
        self, phase: DeliberationPhase, context: Dict
    ) -> "PhaseOutcome":
        """Facilitate a specific phase of deliberation"""
        # Generate phase-specific prompts
        prompts = self._generate_phase_prompts(phase, context)

        # Gather participant contributions
        contributions = await self._gather_phase_contributions(prompts)

        # Process emotional dynamics
        emotional_shifts = await self._process_emotional_dynamics(contributions)

        # Determine phase outcome
        outcome = self._evaluate_phase_outcome(contributions, emotional_shifts)

        return outcome

    async def _process_emotional_dynamics(
        self, contributions: List[Dict[str, Any]]
    ) -> Dict[str, Dict[EmotionalState.Emotion, float]]:
        """Process emotional dynamics from phase contributions.

        Args:
            contributions: List of contributions from the phase

        Returns:
            Dictionary mapping agent IDs to their emotional shifts
        """
        emotional_shifts = {}

        # Group contributions by agent
        agent_contributions = {}
        for contribution in contributions:
            agent_id = contribution["agent_id"]
            if agent_id not in agent_contributions:
                agent_contributions[agent_id] = []
            agent_contributions[agent_id].append(contribution)

        # Process each agent's emotional trajectory
        for agent_id, agent_contribs in agent_contributions.items():
            agent = next(a for a in self.group.agents if a.id == agent_id)

            # Calculate emotional shift based on contributions
            initial_state = agent_contribs[0]["emotional_state"]
            final_state = agent_contribs[-1]["emotional_state"]

            # Initialize shifts dictionary for this agent
            shifts = {emotion: 0.0 for emotion in EmotionalState.Emotion}

            # Calculate emotional shifts
            emotion, intensity = final_state
            base_emotion, base_intensity = initial_state

            # Record the shift in emotion
            shifts[emotion] = intensity - (
                shifts[base_emotion] if base_emotion == emotion else 0.0
            )

            emotional_shifts[agent_id] = shifts

        # Process emotional contagion between agents
        await self._process_emotional_contagion(emotional_shifts)

        return emotional_shifts

    async def _process_emotional_contagion(
        self, emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]]
    ) -> None:
        """Process emotional contagion effects between agents.

        Args:
            emotional_shifts: Current emotional shifts dictionary
        """
        # Calculate influence weights between agents
        influence_weights = {}
        for agent in self.group.agents:
            influence_weights[agent.id] = {}
            for other in self.group.agents:
                if other.id != agent.id:
                    # Calculate influence based on status and relationship
                    influence = (
                        0.4 * agent.status.influence
                        + 0.3 * agent.personality.extraversion
                        + 0.3 * self.group.emotional_network[agent.id][other.id]
                    )
                    influence_weights[agent.id][other.id] = influence

        # Apply emotional contagion effects
        for agent_id, shifts in emotional_shifts.items():
            for other_id, weight in influence_weights[agent_id].items():
                if other_id in emotional_shifts:
                    # Modify other agent's emotions based on influence
                    for emotion in EmotionalState.Emotion:
                        emotional_shifts[other_id][emotion] += (
                            shifts.get(emotion, 0.0) * weight * 0.3  # Contagion factor
                        )

    def _calculate_emotional_influence(
        self, agent1: EnhancedSocialAgent, agent2: EnhancedSocialAgent
    ) -> float:
        """Calculate emotional influence between two agents.

        Args:
            agent1: Source agent (influencer)
            agent2: Target agent (influenced)

        Returns:
            Float between 0 and 1 indicating influence strength
        """
        # Base influence on status and personality
        status_factor = (
            0.4 * agent1.status.influence
            + 0.3 * agent1.status.respect
            + 0.3 * agent1.status.formal_rank
        )

        personality_factor = (
            0.4
            * agent1.personality.extraversion  # More extraverted agents are more influential
            + 0.3
            * agent1.personality.dominance  # More dominant agents have stronger influence
            + 0.3
            * (
                1 - agent2.personality.stability
            )  # Less stable agents are more influenced
        )

        # Consider relationship quality
        relationship_quality = agent1.relationships.get(agent2.id, 0.5)

        # Combine factors with weights
        return (
            0.4 * status_factor + 0.3 * personality_factor + 0.3 * relationship_quality
        )

    def _generate_phase_prompts(
        self, phase: DeliberationPhase, context: Dict
    ) -> List[str]:
        """Generate appropriate prompts for the current deliberation phase.

        Args:
            phase: Current deliberation phase
            context: Current context including situation and participant info

        Returns:
            List of prompts for the current phase
        """
        phase_prompts = {
            self.DeliberationPhase.EMOTIONAL_SHARING: [
                "Share your initial emotional reaction to this situation.",
                "What concerns or hopes does this change bring up for you?",
                "How do you feel this will affect the team dynamics?",
            ],
            self.DeliberationPhase.PERSPECTIVE_BUILDING: [
                "What do you see as the key opportunities in this change?",
                "What potential challenges should we be aware of?",
                "How might this affect different team members differently?",
            ],
            self.DeliberationPhase.SOLUTION_EXPLORATION: [
                "What specific solutions could address the concerns raised?",
                "How might we maximize the potential benefits?",
                "What creative approaches haven't we considered yet?",
            ],
            self.DeliberationPhase.CONSENSUS_BUILDING: [
                "What common ground do you see in our different perspectives?",
                "Which solutions seem to address most of our shared concerns?",
                "What compromises might help us move forward together?",
            ],
            self.DeliberationPhase.COMMITMENT_SECURING: [
                "What support would you need to fully commit to this direction?",
                "How can we ensure everyone's concerns are addressed in implementation?",
                "What role would you like to play in making this successful?",
            ],
        }

        # Add context-specific elements to the prompts
        base_prompts = phase_prompts[phase]
        contextualized_prompts = []
        for prompt in base_prompts:
            contextualized_prompts.append(
                f"Context: {context['situation']}\n"
                f"Stakes: {context['stakes']}\n"
                f"Timeline: {context['timeline']}\n\n"
                f"{prompt}"
            )

        return contextualized_prompts

    async def _gather_phase_contributions(
        self, prompts: List[str]
    ) -> List[Dict[str, Any]]:
        """Gather contributions from all participants for the current phase.

        Args:
            prompts: List of prompts to use for gathering contributions

        Returns:
            List of contribution dictionaries containing responses and metadata
        """
        contributions = []
        for agent in self.group.agents:
            for prompt in prompts:
                contribution = await self._get_agent_contribution(agent, prompt)
                contributions.append(contribution)
        return contributions

    async def _get_agent_contribution(
        self, agent: EnhancedSocialAgent, prompt: str
    ) -> Dict[str, Any]:
        """Get a single agent's contribution in response to a prompt.

        Args:
            agent: The agent providing the contribution
            prompt: The prompt to respond to

        Returns:
            Dictionary containing the contribution and metadata
        """
        # Generate response using the agent's LLM interface
        response = await agent.llm.generate(prompt)

        return {
            "agent_id": agent.id,
            "response": response,
            "emotional_state": agent.emotional_state.get_dominant_emotion(),
            "timestamp": len(self.phase_history),
        }

    def _evaluate_phase_outcome(
        self, contributions: List[Dict], emotional_shifts: Dict
    ) -> "PhaseOutcome":
        """Evaluate the outcome of the current phase."""
        # Calculate consensus level
        consensus_level = self._calculate_consensus_level(contributions)

        # Identify key contributions
        key_contributions = self._identify_key_contributions(contributions)

        # Evaluate emotional progress
        emotional_progress = self._evaluate_emotional_progress(emotional_shifts)

        return PhaseOutcome(
            consensus_level=consensus_level,
            key_contributions=key_contributions,
            emotional_progress=emotional_progress,
        )

    def _calculate_consensus_level(self, contributions: List[Dict]) -> float:
        """Calculate the level of consensus among participants.

        Args:
            contributions: List of all contributions from the phase

        Returns:
            Float between 0 and 1 indicating consensus level
        """
        if not contributions:
            return 0.0

        # Group contributions by agent
        agent_contributions = self._group_contributions_by_agent(contributions)

        # Calculate position similarity between agents
        similarities = []
        agents = list(agent_contributions.keys())

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                similarity = self._calculate_position_similarity(
                    agent_contributions[agents[i]], agent_contributions[agents[j]]
                )
                similarities.append(similarity)

        # Return average similarity if there are any
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_position_similarity(
        self, contributions1: List[Dict], contributions2: List[Dict]
    ) -> float:
        """Calculate similarity between two agents' positions.

        Args:
            contributions1: First agent's contributions
            contributions2: Second agent's contributions

        Returns:
            Float between 0 and 1 indicating position similarity
        """
        # Simple implementation - could be enhanced with NLP
        if not contributions1 or not contributions2:
            return 0.0

        # Compare emotional states
        emotional_similarity = self._calculate_emotional_similarity(
            contributions1[-1]["emotional_state"], contributions2[-1]["emotional_state"]
        )

        # Weight emotional similarity in consensus calculation
        return emotional_similarity

    def _calculate_emotional_similarity(
        self,
        state1: Tuple[EmotionalState.Emotion, float],
        state2: Tuple[EmotionalState.Emotion, float],
    ) -> float:
        """Calculate similarity between emotional states.

        Args:
            state1: First emotional state (emotion, intensity)
            state2: Second emotional state (emotion, intensity)

        Returns:
            Float between 0 and 1 indicating emotional similarity
        """
        emotion1, intensity1 = state1
        emotion2, intensity2 = state2

        # Base similarity on whether emotions match
        emotion_match = 1.0 if emotion1 == emotion2 else 0.0

        # Consider intensity difference
        intensity_similarity = 1.0 - abs(intensity1 - intensity2)

        # Combine both factors
        return 0.7 * emotion_match + 0.3 * intensity_similarity

    def _identify_key_contributions(self, contributions: List[Dict]) -> List[str]:
        """Identify the most significant contributions from the phase.

        Args:
            contributions: List of all contributions

        Returns:
            List of key contribution texts
        """
        if not contributions:
            return []

        # Score contributions based on multiple factors
        scored_contributions = []
        for contribution in contributions:
            score = self._calculate_contribution_significance(contribution)
            scored_contributions.append((score, contribution["response"]))

        # Sort by score and return top contributions
        scored_contributions.sort(reverse=True)
        return [cont for _, cont in scored_contributions[:3]]  # Return top 3

    def _calculate_contribution_significance(self, contribution: Dict) -> float:
        """Calculate the significance score of a contribution.

        Args:
            contribution: Single contribution dictionary

        Returns:
            Float indicating contribution significance
        """
        # Get contributing agent
        agent = next(
            (a for a in self.group.agents if a.id == contribution["agent_id"]), None
        )
        if not agent:
            return 0.0

        # Base score on agent's status and emotional intensity
        status_score = agent.status.compute_effective_status("general")
        _, emotional_intensity = contribution["emotional_state"]

        # Combine factors
        return 0.6 * status_score + 0.4 * emotional_intensity

    def _evaluate_emotional_progress(
        self, emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]]
    ) -> Dict[str, float]:
        """Evaluate emotional progress during the phase.

        Args:
            emotional_shifts: Dictionary of emotional changes

        Returns:
            Dictionary of progress metrics
        """
        if not emotional_shifts:
            return {"overall_progress": 0.0}

        # Calculate various progress metrics
        emotional_alignment = self._calculate_emotional_alignment(emotional_shifts)
        intensity_reduction = self._calculate_intensity_reduction(emotional_shifts)
        positive_shift = self._calculate_positive_shift(emotional_shifts)

        return {
            "emotional_alignment": emotional_alignment,
            "intensity_reduction": intensity_reduction,
            "positive_shift": positive_shift,
            "overall_progress": (
                0.4 * emotional_alignment
                + 0.3 * intensity_reduction
                + 0.3 * positive_shift
            ),
        }

    def _calculate_emotional_alignment(
        self, emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]]
    ) -> float:
        """Calculate how well emotions are aligning between agents."""
        if not emotional_shifts:
            return 0.0

        # Get final emotions for each agent
        final_emotions = {}
        for agent_id, shifts in emotional_shifts.items():
            max_emotion = max(shifts.items(), key=lambda x: x[1])[0]
            final_emotions[agent_id] = max_emotion

        # Calculate proportion of agents sharing the most common emotion
        if not final_emotions:
            return 0.0

        most_common = max(
            set(final_emotions.values()), key=list(final_emotions.values()).count
        )
        return list(final_emotions.values()).count(most_common) / len(final_emotions)

    def _calculate_intensity_reduction(
        self, emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]]
    ) -> float:
        """Calculate reduction in emotional intensity."""
        if not emotional_shifts:
            return 0.0

        # Calculate average intensity change
        intensity_changes = []
        for shifts in emotional_shifts.values():
            initial_intensity = max(shifts.values())
            final_intensity = sum(shifts.values())
            intensity_changes.append(initial_intensity - final_intensity)

        return sum(intensity_changes) / len(intensity_changes)

    def _calculate_positive_shift(
        self, emotional_shifts: Dict[str, Dict[EmotionalState.Emotion, float]]
    ) -> float:
        """Calculate the overall positive emotional shift in the group.

        Args:
            emotional_shifts: Dictionary of emotional changes for each agent

        Returns:
            Float indicating the magnitude of positive shift
        """
        if not emotional_shifts:
            return 0.0

        # Track positive shifts
        positive_shifts = []

        # Calculate positive shift for each agent
        for agent_shifts in emotional_shifts.values():
            # Consider shifts toward positive emotions
            positive_emotions = {
                EmotionalState.Emotion.HAPPY,
                EmotionalState.Emotion.CONFIDENT,
            }

            # Sum shifts toward positive emotions
            positive_magnitude = sum(
                shift
                for emotion, shift in agent_shifts.items()
                if emotion in positive_emotions and shift > 0
            )

            if positive_magnitude > 0:
                positive_shifts.append(positive_magnitude)

        # Return average positive shift
        return sum(positive_shifts) / len(positive_shifts) if positive_shifts else 0.0

    def _group_contributions_by_agent(
        self, contributions: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group contributions by agent ID.

        Args:
            contributions: List of all contributions

        Returns:
            Dictionary mapping agent IDs to their contributions
        """
        grouped = {}
        for contribution in contributions:
            agent_id = contribution["agent_id"]
            if agent_id not in grouped:
                grouped[agent_id] = []
            grouped[agent_id].append(contribution)
        return grouped


class EmotionalLearning:
    """Tracks and applies emotional learning from interactions"""

    def __init__(self, agent: EnhancedSocialAgent):
        self.agent = agent
        self.learned_patterns: Dict[str, float] = {}
        self.adaptation_rate = self._calculate_adaptation_rate()

    def _calculate_adaptation_rate(self) -> float:
        """Calculate how quickly the agent adapts to new emotional patterns.

        Returns:
            Float between 0 and 1 indicating adaptation speed
        """
        # Base rate modified by personality traits
        base_rate = 0.5

        # More open and emotionally stable agents adapt faster
        personality_factor = (
            0.4 * self.agent.personality.openness  # Openness to new patterns
            + 0.3 * self.agent.personality.stability  # Emotional stability
            + 0.3 * self.agent.personality.agreeableness  # Willingness to adjust
        )

        return min(0.9, base_rate + (0.4 * personality_factor))

    async def process_experience(
        self,
        interaction: Union[
            "GroupInteraction", "Conflict", "DecisionOutcome", "AftermathEffect"
        ],
    ) -> None:
        """Process and learn from emotional experiences"""
        # Extract emotional patterns
        patterns = self._identify_emotional_patterns(interaction)

        # Update learned patterns
        self._update_learned_patterns(patterns)

        # Adjust future behavior
        await self._adapt_behavior_patterns()

    def _identify_emotional_patterns(
        self,
        interaction: Union[
            "GroupInteraction", "Conflict", "DecisionOutcome", "AftermathEffect"
        ],
    ) -> Dict[str, float]:
        """Identify emotional patterns from an interaction.

        Args:
            interaction: The interaction to analyze

        Returns:
            Dictionary mapping pattern types to their strengths
        """
        patterns = {}

        if isinstance(interaction, AftermathEffect):
            patterns = self._identify_aftermath_patterns(interaction)
        elif isinstance(interaction, GroupInteraction):
            patterns = self._identify_group_interaction_patterns(interaction)
        elif isinstance(interaction, Conflict):
            patterns = self._identify_conflict_patterns(interaction)
        elif isinstance(interaction, DecisionOutcome):
            patterns = self._identify_decision_patterns(interaction)

        return patterns

    def _identify_aftermath_patterns(
        self, effect: "AftermathEffect"
    ) -> Dict[str, float]:
        """Identify patterns from aftermath effects."""
        patterns = {}

        # Pattern: Response to emotional influence
        if self.agent.id in effect.agents:
            patterns["emotional_influence"] = effect.intensity

        # Pattern: Emotional contagion susceptibility
        if effect.effect_type == EmotionalAftermath.AftermathType.EMOTIONAL_LEARNING:
            patterns["contagion_susceptibility"] = (
                effect.intensity * self.agent.personality.stability
            )

        return patterns

    def _identify_group_interaction_patterns(
        self, interaction: "GroupInteraction"
    ) -> Dict[str, float]:
        """Identify patterns from group interactions."""
        patterns = {}

        # Pattern: Response to interaction types
        patterns["interaction_response"] = self._calculate_interaction_response(
            interaction
        )

        # Pattern: Status influence
        if interaction.initiator.id != self.agent.id:
            patterns["status_influence"] = self._calculate_status_influence(
                interaction.initiator
            )

        return patterns

    def _identify_conflict_patterns(self, conflict: "Conflict") -> Dict[str, float]:
        """Identify patterns from conflicts."""
        patterns = {}

        if self.agent.id in {p.id for p in conflict.participants}:
            # Pattern: Conflict resolution style
            patterns["conflict_style"] = self._calculate_conflict_style(conflict)

            # Pattern: Emotional escalation tendency
            patterns["escalation_tendency"] = self._calculate_escalation_tendency(
                conflict
            )

        return patterns

    def _identify_decision_patterns(
        self, decision: "DecisionOutcome"
    ) -> Dict[str, float]:
        """Identify patterns from decision outcomes."""
        patterns = {}

        # Pattern: Decision acceptance
        patterns["decision_acceptance"] = (
            1.0 if self.agent.id not in decision.dissenting_agents else 0.0
        )

        # Pattern: Consensus alignment
        patterns["consensus_alignment"] = decision.consensus_level

        return patterns

    def _calculate_interaction_response(self, interaction: "GroupInteraction") -> float:
        """Calculate response pattern to interaction types."""
        # Base response on personality traits
        base_response = (
            0.4 * self.agent.personality.extraversion
            + 0.3 * self.agent.personality.openness
            + 0.3 * self.agent.personality.stability
        )

        # Modify based on interaction type
        type_modifiers = {
            GroupInteraction.Type.ANNOUNCEMENT: 0.8,
            GroupInteraction.Type.DISCUSSION: 1.0,
            GroupInteraction.Type.CONFLICT: 0.6,
            GroupInteraction.Type.CELEBRATION: 1.2,
            GroupInteraction.Type.CRISIS: 0.7,
        }

        return base_response * type_modifiers.get(interaction.type, 1.0)

    def _calculate_status_influence(self, other_agent: EnhancedSocialAgent) -> float:
        """Calculate influence of status on emotional response."""
        return (
            0.4 * other_agent.status.influence
            + 0.3 * other_agent.status.respect
            + 0.3 * other_agent.status.formal_rank
        )

    def _calculate_conflict_style(self, conflict: "Conflict") -> float:
        """Calculate conflict handling style tendency."""
        return (
            0.4 * self.agent.personality.dominance
            + 0.3 * self.agent.personality.agreeableness
            + 0.3 * self.agent.personality.stability
        )

    def _calculate_escalation_tendency(self, conflict: "Conflict") -> float:
        """Calculate tendency to escalate conflicts."""
        return (
            0.4 * (1 - self.agent.personality.stability)
            + 0.3 * self.agent.personality.dominance
            + 0.3 * (1 - self.agent.personality.agreeableness)
        )

    def _update_learned_patterns(self, new_patterns: Dict[str, float]) -> None:
        """Update learned patterns with new observations."""
        for pattern_type, strength in new_patterns.items():
            if pattern_type in self.learned_patterns:
                # Update existing pattern with weighted average
                current = self.learned_patterns[pattern_type]
                self.learned_patterns[pattern_type] = (
                    1 - self.adaptation_rate
                ) * current + self.adaptation_rate * strength
            else:
                # Add new pattern
                self.learned_patterns[pattern_type] = strength

    async def _adapt_behavior_patterns(self) -> None:
        """Adapt agent behavior based on learned patterns."""
        # Adjust personality traits within small bounds
        for pattern_type, strength in self.learned_patterns.items():
            if pattern_type == "emotional_influence":
                self._adjust_emotional_sensitivity(strength)
            elif pattern_type == "conflict_style":
                self._adjust_conflict_approach(strength)
            elif pattern_type == "decision_acceptance":
                self._adjust_decision_approach(strength)

    def _adjust_emotional_sensitivity(self, strength: float) -> None:
        """Adjust emotional sensitivity based on learned patterns."""
        # Small adjustment to stability
        current = self.agent.personality.stability
        adjustment = 0.1 * (strength - 0.5)  # Center around neutral
        self.agent.personality.stability = max(0.1, min(0.9, current + adjustment))

    def _adjust_conflict_approach(self, strength: float) -> None:
        """Adjust conflict handling approach."""
        # Small adjustments to dominance and agreeableness
        dom_current = self.agent.personality.dominance
        agree_current = self.agent.personality.agreeableness

        self.agent.personality.dominance = max(
            0.1, min(0.9, dom_current + 0.1 * (strength - 0.5))
        )
        self.agent.personality.agreeableness = max(
            0.1, min(0.9, agree_current + 0.1 * (0.5 - strength))
        )

    def _adjust_decision_approach(self, strength: float) -> None:
        """Adjust decision-making approach."""
        # Small adjustment to openness
        current = self.agent.personality.openness
        adjustment = 0.1 * (strength - 0.5)
        self.agent.personality.openness = max(0.1, min(0.9, current + adjustment))


class AdvancedDeliberationPrompts:
    """Enhanced prompts for sophisticated deliberation"""

    EMOTIONAL_PATTERN_RECOGNITION = """Analyze emotional patterns in group interaction.

Interaction History:
{interaction_details}

Current Emotional States:
{emotional_states}

Relationship Network:
{relationship_network}

Identify:
1. Recurring emotional triggers
2. Response patterns
3. Emotional alliances
4. Trust dynamics
5. Power-emotion interactions

Output format:
{
    "patterns": [
        {
            "type": "pattern_type",
            "description": "pattern_description",
            "participants": ["agent_ids"],
            "strength": float,
            "implications": "future_implications"
        }
    ],
    "recommendations": {
        "facilitation": ["facilitation_strategies"],
        "intervention": ["intervention_points"],
        "support": ["support_needs"]
    }
}"""

    PHASE_TRANSITION = """Guide transition between deliberation phases.

Current Phase: {current_phase}
Next Phase: {next_phase}

Group State:
{group_state}

Emotional Progress:
{emotional_progress}

Generate transition approach that:
1. Acknowledges progress made
2. Prepares group emotionally
3. Sets expectations
4. Maintains momentum
5. Addresses concerns

Consider:
- Emotional readiness
- Group cohesion
- Unresolved tensions
- Learning integration"""


async def run_simulation_scenario():
    """
    Executes a complete social simulation scenario.

    The simulation process:
    1. Initializes agents with distinct personalities
    2. Creates a challenging social situation
    3. Processes individual and group reactions
    4. Facilitates decision-making process
    5. Handles aftermath and learning

    This provides insights into:
    - How personality affects social dynamics
    - How emotions influence group decisions
    - How social hierarchies evolve
    - How relationships develop and change
    """

    # Initialize our cast of characters with distinct personalities and statuses
    agents = {
        "sarah": EnhancedSocialAgent(
            agent_id="sarah",
            llm_model="gpt-4",
            personality=PersonalityTraits(
                extraversion=0.8,
                agreeableness=0.7,
                dominance=0.8,
                openness=0.9,
                stability=0.7,
            ),
            initial_status=SocialStatus(
                formal_rank=0.8,
                influence=0.7,
                respect=0.75,
                expertise={"leadership": 0.8, "communication": 0.9},
            ),
        ),
        "michael": EnhancedSocialAgent(
            agent_id="michael",
            llm_model="gpt-4",
            personality=PersonalityTraits(
                extraversion=0.4,
                agreeableness=0.6,
                dominance=0.3,
                openness=0.5,
                stability=0.9,
            ),
            initial_status=SocialStatus(
                formal_rank=0.6,
                influence=0.5,
                respect=0.7,
                expertise={"technical": 0.9, "planning": 0.8},
            ),
        ),
        "alex": EnhancedSocialAgent(
            agent_id="alex",
            llm_model="gpt-4",
            personality=PersonalityTraits(
                extraversion=0.6,
                agreeableness=0.3,
                dominance=0.7,
                openness=0.4,
                stability=0.5,
            ),
            initial_status=SocialStatus(
                formal_rank=0.7,
                influence=0.6,
                respect=0.7,
                expertise={"technical": 0.8, "communication": 0.7},
            ),
        ),
        "lisa": EnhancedSocialAgent(
            agent_id="lisa",
            llm_model="gpt-4",
            personality=PersonalityTraits(
                extraversion=0.5,
                agreeableness=0.8,
                dominance=0.4,
                openness=0.7,
                stability=0.8,
            ),
            initial_status=SocialStatus(
                formal_rank=0.9,
                influence=0.8,
                respect=0.8,
                expertise={"leadership": 0.9, "communication": 0.8},
            ),
        ),
    }

    # Initialize group dynamics
    group = GroupEmotionalDynamics(list(agents.values()))
    deliberation = AdvancedDeliberation(group)
    aftermath_handler = EmotionalAftermath(group)

    # Create the scenario context
    scenario_context = {
        "situation": "Major team restructuring announcement",
        "stakes": "High",
        "timeline": "Implementation within 2 weeks",
        "initial_positions": {
            "sarah": "Enthusiastic about change, wants to lead implementation",
            "michael": "Concerned about disruption to existing projects",
            "alex": "Resistant to change, fears loss of influence",
            "lisa": "Neutral but worried about team harmony",
        },
    }

    # Start the simulation
    print("=== Starting Simulation ===\n")

    # Phase 1: Initial Reactions
    print("=== Phase 1: Initial Reactions ===")
    initial_interaction = GroupInteraction(
        interaction_type=GroupInteraction.Type.ANNOUNCEMENT,
        initiator=agents["sarah"],
        participants=set(agents.values()),
        context=scenario_context["situation"],
    )

    emotional_responses = await group.process_group_interaction(initial_interaction)
    print_emotional_state(emotional_responses)

    # Phase 2: Deliberation Process
    print("\n=== Phase 2: Deliberation Process ===")
    for phase in AdvancedDeliberation.DeliberationPhase:
        phase_outcome = await deliberation.facilitate_phase(phase, scenario_context)
        print_phase_outcome(phase, phase_outcome)

    # Create decision proposal
    proposal = DecisionProposal(
        topic="Team Restructuring Implementation",
        options=[
            "Immediate full reorganization",
            "Phased transition over 1 month",
            "Hybrid approach with pilot team",
        ],
        initiator=agents["sarah"],
        context=scenario_context["situation"],
        urgency=0.8,
    )

    # Phase 3: Decision Making
    print("\n=== Phase 3: Decision Making ===")
    decision_maker = EmotionalDecisionMaking(group)
    decision = await decision_maker.make_group_decision(proposal)
    print_decision_outcome(decision)

    # Phase 4: Aftermath
    print("\n=== Phase 4: Emotional Aftermath ===")
    await aftermath_handler.process_aftermath(decision)
    final_state = group.analyze_hierarchy()
    print_final_state(final_state)


def print_emotional_state(responses: Optional[Dict[str, EmotionalState]]) -> None:
    """Pretty print emotional responses"""
    if not responses:
        print("No emotional responses to display")
        return

    for agent_id, state in responses.items():
        try:
            emotion, intensity = state.get_dominant_emotion()
            print(
                f"{agent_id.capitalize()}: {emotion.value} (intensity: {intensity:.2f})"
            )
        except Exception as e:
            print(f"Error processing {agent_id}'s emotional state: {e}")


def print_phase_outcome(phase, outcome):
    """Pretty print phase outcomes"""
    print(f"\nPhase: {phase.value}")
    print(f"Consensus Level: {outcome.consensus_level:.2f}")
    print("Key Contributions:")
    for contribution in outcome.key_contributions[:2]:  # Show top 2 contributions
        print(f"- {contribution}")


def print_decision_outcome(decision: DecisionOutcome) -> None:
    """Pretty print decision outcome"""
    print(f"Decision: {decision.decision}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Consensus Level: {decision.consensus_level:.2f}")
    print("\nEmotional Impact:")
    for key, value in decision.emotional_impact.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.2f}")
        else:
            print(f"- {key}: {value}")
    print("\nDissenting Agents:")
    for agent_id in decision.dissenting_agents:
        print(f"- {agent_id}")


def print_final_state(state):
    """Pretty print final emotional and social state"""
    print("Final Group State:")
    print(f"Cohesion Level: {state['cohesion_level']:.2f}")
    print("\nRelationship Changes:")
    for change in state["relationship_changes"][:3]:  # Show top 3 changes
        print(f"- {change}")
    print("\nEmergent Leaders:")
    for leader in state["emergent_leaders"]:
        print(f"- {leader}")


# Run the simulation
if __name__ == "__main__":
    asyncio.run(run_simulation_scenario())
