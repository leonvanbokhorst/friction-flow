from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

class AgentConfig(BaseModel):
    """Configuration for individual agents"""
    
    personality_variance: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Variance in personality traits between agents"
    )
    emotional_decay_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Rate at which emotions decay over time"
    )
    interaction_radius: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Social distance within which agents can interact"
    )
    memory_span: int = Field(
        default=100,
        ge=1,
        description="Number of past interactions to remember"
    )

class GroupConfig(BaseModel):
    """Configuration for group dynamics"""
    
    min_group_size: int = Field(
        default=3,
        ge=2,
        description="Minimum number of agents to form a group"
    )
    cohesion_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum cohesion level for group formation"
    )
    max_groups: int = Field(
        default=10,
        ge=1,
        description="Maximum number of groups allowed"
    )

class SimulationConfig(BaseModel):
    """Main simulation configuration"""
    
    num_agents: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Number of agents in the simulation"
    )
    duration: int = Field(
        default=1000,
        ge=1,
        description="Number of simulation steps to run"
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    agent_config: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Configuration for agents"
    )
    group_config: GroupConfig = Field(
        default_factory=GroupConfig,
        description="Configuration for group dynamics"
    )
    
    @classmethod
    def from_yaml(cls, path: Path) -> "SimulationConfig":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.parse_obj(config_dict) 