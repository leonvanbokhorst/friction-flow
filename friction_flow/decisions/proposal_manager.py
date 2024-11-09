from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from uuid import uuid4

@dataclass
class Proposal:
    """Represents a proposal in the deliberation process"""
    
    id: str
    author_id: str
    content: Dict[str, Any]
    support_count: int = 0
    opposition_count: int = 0
    
    @property
    def approval_ratio(self) -> float:
        """Calculate the ratio of support to total votes"""
        total = self.support_count + self.opposition_count
        return self.support_count / total if total > 0 else 0.0

class ProposalManager:
    """Manages proposals during deliberation"""
    
    def __init__(self):
        self.active_proposals: Dict[str, Proposal] = {}
        self.archived_proposals: List[Proposal] = []
    
    def create_proposal(
        self,
        author_id: str,
        content: Dict[str, Any]
    ) -> Proposal:
        """Create and register a new proposal"""
        proposal = Proposal(
            id=str(uuid4()),
            author_id=author_id,
            content=content
        )
        self.active_proposals[proposal.id] = proposal
        return proposal
    
    def vote_on_proposal(
        self,
        proposal_id: str,
        support: bool
    ) -> None:
        """Record a vote on a proposal"""
        if proposal_id in self.active_proposals:
            proposal = self.active_proposals[proposal_id]
            if support:
                proposal.support_count += 1
            else:
                proposal.opposition_count += 1
    
    def get_winning_proposal(self) -> Optional[Proposal]:
        """Get the proposal with the highest approval ratio"""
        if not self.active_proposals:
            return None
            
        return max(
            self.active_proposals.values(),
            key=lambda p: p.approval_ratio
        ) 