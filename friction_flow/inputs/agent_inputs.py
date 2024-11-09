from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import numpy as np

class PersonalityTraits(BaseModel):
    """Big Five personality traits plus additional social traits"""
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)
    social_influence: float = Field(ge=0.0, le=1.0)
    
    @classmethod
    def random(cls) -> "PersonalityTraits":
        """Generate random personality traits"""
        return cls(
            openness=np.random.beta(2, 2),
            conscientiousness=np.random.beta(2, 2),
            extraversion=np.random.beta(2, 2),
            agreeableness=np.random.beta(2, 2),
            neuroticism=np.random.beta(2, 2),
            dominance=np.random.beta(2, 2),
            social_influence=np.random.beta(2, 2)
        )

class EmotionalTrait(BaseModel):
    """Emotional characteristics of an agent"""
    baseline_mood: float = Field(ge=-1.0, le=1.0)
    emotional_volatility: float = Field(ge=0.0, le=1.0)
    emotional_resilience: float = Field(ge=0.0, le=1.0)
    empathy: float = Field(ge=0.0, le=1.0)
    
    @classmethod
    def random(cls) -> "EmotionalTrait":
        return cls(
            baseline_mood=np.random.normal(0.2, 0.3),  # Slightly positive bias
            emotional_volatility=np.random.beta(2, 3),  # More stable than volatile
            emotional_resilience=np.random.beta(3, 2),  # More resilient than not
            empathy=np.random.beta(2, 2)  # Bell curve distribution
        )

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
            "emotional_expression": 0.5
        }
    )

class GroupPreference(BaseModel):
    """Defines initial group preferences"""
    agent_id: str
    preferred_partners: List[str]
    avoided_partners: List[str] = []
    
    @validator('preferred_partners', 'avoided_partners')
    def unique_partners(cls, v):
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
    
    @validator('relationships')
    def validate_relationships(cls, relationships, values):
        """Ensure relationships reference valid agents"""
        if 'agents' not in values:
            return relationships
            
        agent_ids = {agent.id for agent in values['agents']}
        for rel in relationships:
            if rel.agent_a not in agent_ids:
                raise ValueError(f"Agent {rel.agent_a} not found")
            if rel.agent_b not in agent_ids:
                raise ValueError(f"Agent {rel.agent_b} not found")
        return relationships
    
    @classmethod
    def generate_random(cls, num_agents: int) -> "SimulationInputs":
        """Generate random simulation inputs"""
        # Ensure one leader and appropriate distribution of roles
        roles = [SocialRole.LEADER]  # One leader
        roles.extend([SocialRole.INFLUENCER] * int(num_agents * 0.1))  # 10% influencers
        roles.extend([SocialRole.MEDIATOR] * int(num_agents * 0.2))    # 20% mediators
        roles.extend([SocialRole.OBSERVER] * int(num_agents * 0.1))    # 10% observers
        # Fill the rest with followers
        roles.extend([SocialRole.FOLLOWER] * (num_agents - len(roles)))
        
        # Shuffle roles
        np.random.shuffle(roles)
        
        agents = [
            AgentInput(
                id=f"agent_{i}",
                name=f"Agent {i}",
                personality=PersonalityTraits.random(),
                emotional_traits=EmotionalTrait.random(),
                initial_status=np.random.beta(2, 2),
                role=roles[i],  # Now passing proper SocialRole enum instance
                conflict_tolerance=np.random.beta(2, 2),
                change_resistance=np.random.beta(2, 2),
                preferred_group_size=np.random.randint(2, 10),
                communication_style={
                    "directness": np.random.beta(2, 2),
                    "formality": np.random.beta(2, 2),
                    "emotional_expression": np.random.beta(2, 2)
                }
            )
            for i in range(num_agents)
        ]
        
        # Generate relationships
        relationships = []
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if np.random.random() < 0.3:  # 30% chance of relationship
                    relationships.append(
                        InitialRelationship(
                            agent_a=f"agent_{i}",
                            agent_b=f"agent_{j}",
                            trust=np.random.beta(2, 2),
                            familiarity=np.random.beta(2, 2),
                            emotional_bond=np.random.beta(2, 2)
                        )
                    )
        
        return cls(
            agents=agents,
            relationships=relationships,
            environmental_stress=np.random.beta(2, 2),
            communication_noise=np.random.beta(2, 2),
            resource_scarcity=np.random.beta(2, 2)
        ) 