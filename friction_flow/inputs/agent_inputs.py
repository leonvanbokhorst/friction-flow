from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator, model_validator, field_validator
import numpy as np
import logging
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PersonalityTraits(BaseModel):
    """Big Five personality traits plus additional social traits"""

    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)
    social_influence: float = Field(ge=0.0, le=1.0)

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }

    def model_dump(self, **kwargs) -> dict:
        """Ensure all fields are included in serialization in a consistent order"""
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism,
            'dominance': self.dominance,
            'social_influence': self.social_influence
        }


class EmotionalTrait(BaseModel):
    """Emotional characteristics of an agent"""

    baseline_mood: float = Field(ge=-1.0, le=1.0)
    emotional_volatility: float = Field(ge=0.0, le=1.0)
    emotional_resilience: float = Field(ge=0.0, le=1.0)
    empathy: float = Field(ge=0.0, le=1.0)

    @classmethod
    def random(cls) -> "EmotionalTrait":
        """Generate random emotional traits"""
        # Clip baseline_mood to ensure it stays within [-1, 1]
        return cls(
            baseline_mood=float(np.clip(np.random.normal(0.2, 0.3), -1.0, 1.0)),
            emotional_volatility=float(np.random.beta(2, 2)),
            emotional_resilience=float(np.random.beta(2, 2)),
            empathy=float(np.random.beta(2, 2))
        )

    def model_dump(self, **kwargs) -> dict:
        """Ensure proper serialization"""
        return {
            'baseline_mood': float(self.baseline_mood),
            'emotional_volatility': float(self.emotional_volatility),
            'emotional_resilience': float(self.emotional_resilience),
            'empathy': float(self.empathy)
        }


class SocialRole(str, Enum):
    LEADER = "leader"
    INFLUENCER = "influencer"
    MEDIATOR = "mediator"
    FOLLOWER = "follower"
    OBSERVER = "observer"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: Any) -> "SocialRole":
        """Handle missing values by trying to match strings"""
        if isinstance(value, str):
            # Try to match case-insensitive
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return super()._missing_(value)


class InitialRelationship(BaseModel):
    """Defines initial relationship between two agents"""

    agent_a: str
    agent_b: str
    trust: float = Field(ge=0.0, le=1.0)
    familiarity: float = Field(ge=0.0, le=1.0)
    emotional_bond: float = Field(ge=0.0, le=1.0)


class AgentInput(BaseModel):
    """Enhanced input definition for a single agent"""

    id: str
    name: str
    personality: PersonalityTraits
    emotional_traits: EmotionalTrait
    initial_status: float = Field(ge=0.0, le=1.0, default=0.5)
    role: SocialRole = SocialRole.FOLLOWER

    # Social preferences
    preferred_group_size: int = Field(ge=2, le=10, default=5)
    conflict_tolerance: float = Field(ge=0.0, le=1.0)
    change_resistance: float = Field(ge=0.0, le=1.0)

    # Communication style
    communication_style: Dict[str, float] = Field(
        default_factory=lambda: {
            "directness": 0.5,
            "formality": 0.5,
            "emotional_expression": 0.5,
            "assertiveness": 0.5,
            "cooperation": 0.5,
        }
    )

    @classmethod
    def model_validate(cls, obj: Any) -> "AgentInput":
        """Custom validation for loading from dict/yaml"""
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict, got {type(obj)}")
            
        logger.debug(f"Validating agent input: {obj.get('id', 'NO_ID')}")
        
        # Explicitly validate personality traits
        if 'personality' in obj:
            personality_data = obj['personality']
            logger.debug(f"Raw personality data: {personality_data}")
            
            # Let Pydantic handle the validation through PersonalityTraits
            try:
                obj['personality'] = PersonalityTraits.model_validate(personality_data)
                logger.debug(f"Created personality traits: {obj['personality'].model_dump()}")
            except Exception as e:
                logger.error(f"Error creating personality traits: {e}")
                raise
        
        return super().model_validate(obj)

    def model_dump(self, **kwargs) -> dict:
        """Ensure proper serialization of nested models"""
        logger.debug(f"AgentInput.model_dump called for agent {self.id}")
        
        try:
            # Get personality data without additional validation
            personality_data = self.personality.model_dump(**kwargs)
            
            # Create the full agent data
            data = {
                'id': self.id,
                'name': self.name,
                'personality': personality_data,
                'emotional_traits': self.emotional_traits.model_dump(**kwargs),
                'initial_status': float(self.initial_status),
                'role': str(self.role),
                'preferred_group_size': int(self.preferred_group_size),
                'conflict_tolerance': float(self.conflict_tolerance),
                'change_resistance': float(self.change_resistance),
                'communication_style': {
                    k: float(v) for k, v in self.communication_style.items()
                }
            }
            
            logger.debug(f"Final agent data: {data}")
            return data
            
        except Exception as e:
            logger.error(f"Error in AgentInput.model_dump: {e}")
            logger.error(f"Agent ID: {self.id}")
            logger.error(f"Current personality state: {self.personality.__dict__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


class GroupPreference(BaseModel):
    """Defines initial group preferences"""

    agent_id: str
    preferred_partners: List[str]
    avoided_partners: List[str] = []

    @field_validator("preferred_partners", "avoided_partners")
    @classmethod
    def unique_partners(cls, v: List[str]) -> List[str]:
        if len(set(v)) != len(v):
            raise ValueError("Duplicate partners in preferences")
        return v


class SimulationInputs(BaseModel):
    """Enhanced input definition for simulation"""

    agents: List[AgentInput]
    relationships: List[InitialRelationship] = []
    group_preferences: List[GroupPreference] = []

    # Environmental factors
    environmental_stress: float = Field(ge=0.0, le=1.0, default=0.0)
    communication_noise: float = Field(ge=0.0, le=1.0, default=0.0)
    resource_scarcity: float = Field(ge=0.0, le=1.0, default=0.0)

    @model_validator(mode="after")
    def validate_relationships(self) -> "SimulationInputs":
        """Ensure relationships reference valid agents"""
        agent_ids = {agent.id for agent in self.agents}

        for rel in self.relationships:
            if rel.agent_a not in agent_ids:
                raise ValueError(f"Agent {rel.agent_a} not found")
            if rel.agent_b not in agent_ids:
                raise ValueError(f"Agent {rel.agent_b} not found")

        return self

    @model_validator(mode="after")
    def validate_group_preferences(self) -> "SimulationInputs":
        """Ensure group preferences reference valid agents"""
        agent_ids = {agent.id for agent in self.agents}

        for pref in self.group_preferences:
            if pref.agent_id not in agent_ids:
                raise ValueError(
                    f"Agent {pref.agent_id} not found in group preferences"
                )

        return self

    @field_validator("environmental_stress", "communication_noise", "resource_scarcity")
    @classmethod
    def validate_environmental_factors(cls, v: float, info: Any) -> float:
        """Ensure environmental factors are properly scaled"""
        if v > 0.8:
            logger.warning(f"High {info.field_name}: {v}")
        return v

    @classmethod
    def generate_random(cls, num_agents: int) -> "SimulationInputs":
        """Generate random simulation inputs with realistic distributions"""
        try:
            # Generate roles with proper distribution
            roles = [SocialRole.LEADER]  # One leader
            roles.extend(
                [SocialRole.INFLUENCER] * int(num_agents * 0.1)
            )  # 10% influencers
            roles.extend([SocialRole.MEDIATOR] * int(num_agents * 0.2))  # 20% mediators
            roles.extend([SocialRole.OBSERVER] * int(num_agents * 0.1))  # 10% observers
            roles.extend([SocialRole.FOLLOWER] * (num_agents - len(roles)))
            np.random.shuffle(roles)

            agents = []
            for i in range(num_agents):
                try:
                    # Create base traits with explicit logging
                    base_traits = {
                        "openness": float(np.random.beta(2, 2)),
                        "conscientiousness": float(np.random.beta(2, 2)),
                        "extraversion": float(np.random.beta(2, 2)),
                        "agreeableness": float(np.random.beta(2, 2)),
                        "neuroticism": float(np.random.beta(2, 2)),
                        "dominance": float(np.random.beta(2, 2)),
                        "social_influence": float(np.random.beta(2, 2))
                    }
                    
                    logger.debug(f"Creating personality for agent {i} with base traits: {base_traits}")
                    
                    # Create personality object
                    personality = PersonalityTraits(**base_traits)
                    
                    # Log the personality object state
                    logger.debug(f"Initial personality state for agent {i}: {personality.model_dump()}")
                    
                    # Modify traits based on role
                    if roles[i] in [SocialRole.LEADER, SocialRole.INFLUENCER]:
                        new_dominance = float(np.clip(np.random.beta(3, 2), 0, 1))
                        new_influence = float(np.clip(np.random.beta(3, 2), 0, 1))
                        logger.debug(f"Modifying traits for {roles[i]}: dominance={new_dominance}, influence={new_influence}")
                        
                        personality.dominance = new_dominance
                        personality.social_influence = new_influence
                        
                        # Log after modification
                        logger.debug(f"Modified personality state: {personality.model_dump()}")

                    # Create agent with detailed logging
                    logger.debug(f"Creating agent {i} with role {roles[i]}")
                    agent = AgentInput(
                        id=f"agent_{i}",
                        name=f"Agent {i}",
                        personality=personality,
                        emotional_traits=EmotionalTrait.random(),
                        initial_status=(
                            float(np.random.beta(3, 2))
                            if roles[i] in [SocialRole.LEADER, SocialRole.INFLUENCER]
                            else float(np.random.beta(2, 2))
                        ),
                        role=roles[i],
                        conflict_tolerance=float(np.random.beta(2, 2)),
                        change_resistance=float(np.random.beta(2, 2)),
                        preferred_group_size=int(np.random.randint(2, 10)),
                        communication_style={
                            "directness": float(np.random.beta(2, 2)),
                            "formality": float(np.random.beta(2, 2)),
                            "emotional_expression": float(np.random.beta(2, 2)),
                            "assertiveness": float(np.random.beta(2, 2)),
                            "cooperation": float(np.random.beta(2, 2)),
                        },
                    )
                    
                    # Verify agent creation
                    logger.debug(f"Created agent {i}. Verifying personality traits...")
                    agent_data = agent.model_dump()
                    logger.debug(f"Agent {i} personality after dump: {agent_data['personality']}")
                    
                    agents.append(agent)

                except Exception as e:
                    logger.error(f"Error creating agent {i}:")
                    logger.error(f"Last known base traits: {base_traits}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise

            # Create simulation with logging
            logger.debug(f"Creating simulation with {len(agents)} agents")
            simulation = cls(
                agents=agents,
                environmental_stress=float(np.random.beta(2, 2)),
                communication_noise=float(np.random.beta(2, 2)),
                resource_scarcity=float(np.random.beta(2, 2)),
            )

            # Generate relationships with realistic network properties
            relationships = []
            for i in range(num_agents):
                # Create more connections for leaders and influencers
                num_connections = int(np.random.beta(2, 2) * num_agents * 0.3)
                if roles[i] in [SocialRole.LEADER, SocialRole.INFLUENCER]:
                    num_connections = int(num_connections * 1.5)

                possible_connections = list(range(num_agents))
                possible_connections.remove(i)
                connections = np.random.choice(
                    possible_connections,
                    size=min(num_connections, len(possible_connections)),
                    replace=False,
                )

                for j in connections:
                    # Base relationship strength on personality compatibility
                    compatibility = 1 - abs(
                        agents[i].personality.agreeableness
                        - agents[j].personality.agreeableness
                    )

                    relationships.append(
                        InitialRelationship(
                            agent_a=f"agent_{i}",
                            agent_b=f"agent_{j}",
                            trust=np.random.beta(2, 2) * compatibility,
                            familiarity=np.random.beta(2, 2),
                            emotional_bond=np.random.beta(2, 2) * compatibility,
                        )
                    )

            return simulation

        except Exception as e:
            logger.error("Error in generate_random:")
            logger.error(traceback.format_exc())
            raise

    def model_dump(self, **kwargs) -> dict:
        """Ensure proper serialization of nested models"""
        data = super().model_dump(**kwargs)
        # Ensure agents are fully serialized
        if 'agents' in data:
            data['agents'] = [agent.model_dump(**kwargs) for agent in self.agents]
        return data
