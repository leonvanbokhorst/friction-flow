from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio

from ..core.base_classes import Observable
from ..agents.enhanced_social_agent import EnhancedSocialAgent
from ..emotions.emotional_state import EmotionalState

class DeliberationPhase(Enum):
    PROPOSAL = "proposal"
    DISCUSSION = "discussion"
    VOTING = "voting"
    RESOLUTION = "resolution"

@dataclass
class DeliberationContext:
    """Stores the context for a deliberation session"""
    topic: str
    current_phase: DeliberationPhase
    proposals: List[Dict[str, Any]]
    votes: Dict[str, str]  # agent_id -> proposal_id
    discussion_history: List[Dict[str, Any]]
    
class GroupDeliberation(Observable):
    """Manages the group deliberation process"""
    
    def __init__(self, agents: List[EnhancedSocialAgent]):
        super().__init__()
        self.agents = agents
        self.current_context: Optional[DeliberationContext] = None
        self.phase_duration = {
            DeliberationPhase.PROPOSAL: 300,    # 5 minutes
            DeliberationPhase.DISCUSSION: 600,  # 10 minutes
            DeliberationPhase.VOTING: 300,      # 5 minutes
            DeliberationPhase.RESOLUTION: 120   # 2 minutes
        }
    
    async def start_deliberation(self, topic: str) -> None:
        """Initialize a new deliberation session"""
        self.current_context = DeliberationContext(
            topic=topic,
            current_phase=DeliberationPhase.PROPOSAL,
            proposals=[],
            votes={},
            discussion_history=[]
        )
        await self._run_deliberation_cycle()
    
    async def _run_deliberation_cycle(self) -> None:
        """Run through all phases of deliberation"""
        for phase in DeliberationPhase:
            self.current_context.current_phase = phase
            self._notify_observers({
                'type': 'phase_change',
                'phase': phase,
                'topic': self.current_context.topic
            })
            
            await asyncio.sleep(self.phase_duration[phase])
            await self._process_phase(phase)
    
    async def _process_phase(self, phase: DeliberationPhase) -> None:
        """Process the current deliberation phase"""
        if phase == DeliberationPhase.PROPOSAL:
            await self._collect_proposals()
        elif phase == DeliberationPhase.DISCUSSION:
            await self._facilitate_discussion()
        elif phase == DeliberationPhase.VOTING:
            await self._collect_votes()
        else:  # RESOLUTION
            await self._resolve_deliberation() 