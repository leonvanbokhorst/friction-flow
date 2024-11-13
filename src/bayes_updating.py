"""
Bayesian Belief Network for Dynamic Belief Updating Using Language Models
======================================================================

This module implements a novel approach to modeling and updating belief systems using
language model embeddings and Bayesian inference. It demonstrates how beliefs about
topics can evolve based on new evidence while maintaining uncertainty estimates.

Core Concepts:
-------------
1. Belief Representation:
   - Beliefs are represented as high-dimensional vectors in embedding space
   - Each dimension captures semantic aspects of the belief
   - Vectors are generated using LLM embeddings for consistent semantic meaning

2. Bayesian Framework:
   - Prior: Current belief state and confidence
   - Likelihood: Similarity between new evidence and current belief
   - Posterior: Updated belief incorporating new evidence
   - Confidence: Uncertainty measure updated via Bayes' rule

3. Belief Evolution:
   - Beliefs change gradually through weighted averaging
   - Confidence levels affect the impact of new evidence
   - Historical states are maintained for analysis
   - Time-based decay models forgetting and uncertainty growth

Key Components:
--------------
- BeliefState: Data structure holding current beliefs and history
- BayesianBeliefUpdater: Core logic for belief updates
- BeliefVisualizer: Visualization of belief evolution

Example Experiment:
-----------------
The main() function demonstrates the system using AI ethics as a test domain:
1. Starts with neutral belief about AI ethics
2. Processes sequence of statements representing different viewpoints
3. Shows how beliefs evolve from:
   - Initial safety-focused perspective
   - Through various challenging viewpoints
   - To more nuanced understanding incorporating multiple aspects
4. Demonstrates confidence dynamics as:
   - Increases with confirming evidence
   - Decreases with contradictory evidence
   - Decays over time without updates

Usage:
------

```python
llm = OllamaInterface()
updater = BayesianBeliefUpdater(llm)
await updater.initialize_belief_state("AI ethics")
new_state = await updater.update_belief("AI ethics", "New evidence...")
analysis = await updater.analyze_belief_shift("AI ethics")
```

Mathematical Foundation:
----------------------
The system implements Bayes' theorem:
P(belief|evidence) ∝ P(evidence|belief) * P(belief)

Where:
- P(belief) is the prior confidence
- P(evidence|belief) is calculated via cosine similarity
- P(belief|evidence) becomes the posterior confidence

The belief vector itself is updated using weighted averaging:
new_belief = (1-w) * old_belief + w * evidence
where w is derived from the posterior confidence
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from language_models import LanguageModel, OllamaInterface
import logging
from logging_config import setup_logging
from belief_visualizer import BeliefVisualizer

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class BeliefState:
    """
    A data structure that represents a belief about a topic using vector embeddings.

    The belief state combines two key aspects:
    1. Semantic Understanding: High-dimensional vector capturing meaning
    2. Epistemic Uncertainty: Confidence level in current belief

    Key Components:
    --------------
    belief_vector : np.ndarray
        High-dimensional embedding vector representing the semantic content of the belief.
        - Each dimension captures different aspects of meaning
        - Normalized to unit length for consistent comparison
        - Generated from LLM embeddings

    confidence : float
        Scalar value [0-1] indicating certainty in current belief
        - 0.0 = complete uncertainty
        - 1.0 = absolute certainty (never actually reached)
        - Decays over time without reinforcement

    prior_states : List[Tuple[np.ndarray, float]]
        Historical record of previous beliefs and confidences
        - Enables analysis of belief evolution
        - Limited by max_history to prevent unbounded growth
        - Used for visualization and trend analysis

    themes : List[str]
        Identified themes in the belief content
        - Currently unused but prepared for future theme tracking
        - Will enable analysis of belief clusters and patterns
    """

    belief_vector: np.ndarray
    confidence: float
    max_history: int = 100  # Default max history size
    prior_states: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure prior_states doesn't exceed max_history"""
        if len(self.prior_states) > self.max_history:
            self.prior_states = self.prior_states[-self.max_history :]


class BayesianBeliefUpdater:
    """
    Implements dynamic belief updating using Bayesian inference and LLM embeddings.

    Core Algorithm:
    --------------
    1. Belief Representation:
       - Uses LLM embeddings to capture semantic meaning
       - Maintains normalized vectors for consistent comparison
       - Tracks confidence separately from belief content

    2. Update Mechanism:
       a) Prior Capture:
          - Stores current state before update
          - Maintains limited history

       b) Evidence Processing:
          - Converts new evidence to embedding
          - Ensures consistent semantic space

       c) Likelihood Calculation:
          - Uses cosine similarity
          - Higher similarity = stronger support for current belief

       d) Confidence Update:
          - Applies Bayes' rule
          - Includes time-based decay
          - More sensitive to contradictory evidence

       e) Belief Vector Update:
          - Weighted average based on confidence
          - Ensures smooth transitions
          - Maintains vector normalization

    Design Principles:
    -----------------
    1. Conservative Updates:
       - Beliefs change gradually
       - Requires consistent evidence for major shifts

    2. Uncertainty Handling:
       - Confidence decays over time
       - Contradictory evidence reduces confidence faster
       - Maximum confidence is capped

    3. Memory Effects:
       - Maintains history of belief states
       - Enables analysis of belief evolution
       - Supports visualization of changes
    """

    def __init__(self, llm: LanguageModel):
        self.llm = llm
        self.belief_states: Dict[str, BeliefState] = {}
        self.theme_weights = {}
        self.logger = logging.getLogger(__name__)

    async def initialize_belief_state(
        self, topic: str, max_history: int = 100
    ) -> BeliefState:
        """
        Initialize a new belief state for a given topic.

        Args:
            topic: The topic to initialize beliefs for
            max_history: Maximum number of historical states to maintain
        """
        embedding = await self.llm.generate_embedding(topic)

        belief_state = BeliefState(
            belief_vector=np.array(embedding),
            confidence=0.5,  # Start with moderate confidence
            max_history=max_history,
            themes=[],
        )

        self.belief_states[topic] = belief_state
        return belief_state

    async def update_belief(self, topic: str, new_evidence: str) -> BeliefState:
        """
        The core belief updating algorithm:

        1. Prior Capture:
           - Stores current belief state before update
           - Preserves history for analysis

        2. Evidence Processing:
           - Converts new text evidence into embedding vector
           - Uses LLM to generate semantic representation

        3. Likelihood Calculation:
           - Uses cosine similarity between current belief and new evidence
           - Higher similarity = higher likelihood of evidence supporting current belief

        4. Bayesian Update:
           - P(belief|evidence) ∝ P(evidence|belief) * P(belief)
           - Updates confidence using Bayes' rule

        5. Belief Vector Update:
           - Performs weighted average between old and new beliefs
           - Weight determined by posterior confidence
           - Ensures smooth belief transitions

        6. Normalization:
           - Maintains unit vector for consistent comparisons
        """
        if topic not in self.belief_states:
            await self.initialize_belief_state(topic)

        current_state = self.belief_states[topic]

        # Store current state in history with size limit
        current_state.prior_states.append(
            (current_state.belief_vector.copy(), current_state.confidence)
        )
        if len(current_state.prior_states) > current_state.max_history:
            current_state.prior_states = current_state.prior_states[
                -current_state.max_history :
            ]

        # Generate embedding for new evidence
        evidence_embedding = np.array(await self.llm.generate_embedding(new_evidence))

        # Calculate likelihood using cosine similarity
        likelihood = np.dot(current_state.belief_vector, evidence_embedding) / (
            np.linalg.norm(current_state.belief_vector)
            * np.linalg.norm(evidence_embedding)
        )

        # Confidence updating with better dampening
        confidence_decay_factor = 0.995
        posterior_confidence = min(
            0.95,  # Maximum confidence cap
            (current_state.confidence * likelihood * confidence_decay_factor)
            / (
                (current_state.confidence * likelihood)
                + (1 - current_state.confidence) * (1 - likelihood)
            ),
        )

        # More sensitive to divergent evidence
        if likelihood < 0.7:  # Threshold for divergent evidence
            confidence_decay = 0.8
            posterior_confidence *= confidence_decay

        # Time-based decay
        time_decay = 0.99 ** len(current_state.prior_states)
        posterior_confidence *= time_decay

        # Update belief vector using weighted average
        weight = posterior_confidence / (
            current_state.confidence + posterior_confidence
        )
        updated_belief = (
            1 - weight
        ) * current_state.belief_vector + weight * evidence_embedding

        # Normalize updated belief vector
        updated_belief = updated_belief / np.linalg.norm(updated_belief)

        # Update state
        current_state.belief_vector = updated_belief
        current_state.confidence = posterior_confidence

        return current_state

    async def analyze_belief_shift(self, topic: str) -> Dict:
        """
        Analyzes the evolution of beliefs over time:

        1. Tracks sequential changes in belief vectors
        2. Measures magnitude of belief shifts using cosine similarity
        3. Monitors confidence changes
        4. Returns structured analysis of belief evolution

        Used for:
        - Understanding how beliefs evolve
        - Detecting significant opinion changes
        - Monitoring confidence dynamics
        """
        if topic not in self.belief_states:
            return {"error": "Topic not found"}

        state = self.belief_states[topic]
        shifts = []

        for i, (prior_vector, prior_conf) in enumerate(state.prior_states):
            if i > 0:
                shift = np.dot(prior_vector, state.prior_states[i - 1][0]) / (
                    np.linalg.norm(prior_vector)
                    * np.linalg.norm(state.prior_states[i - 1][0])
                )
                shifts.append(
                    {
                        "step": i,
                        "shift_magnitude": float(shift),
                        "confidence_change": float(
                            prior_conf - state.prior_states[i - 1][1]
                        ),
                    }
                )

        return {
            "total_updates": len(state.prior_states),
            "current_confidence": float(state.confidence),
            "belief_shifts": shifts,
        }


async def main():
    """Main execution function for demonstration."""
    # Initialize
    llm = OllamaInterface()
    belief_updater = BayesianBeliefUpdater(llm)
    visualizer = BeliefVisualizer()

    # Initialize belief state for a topic
    topic = "AI ethics"
    await belief_updater.initialize_belief_state(topic)

    evidence_list = [
        "AI ethics is fundamentally about ensuring responsible development",
        "AI systems should prioritize human values and safety",
        "Recent AI advances show concerning emergent behaviors",
        "AI alignment may be impossible to fully achieve",
        "Open source AI development reduces existential risks",
        "AI systems demonstrate unexpected beneficial behaviors",
        "Regulation could increase AI safety risks",
        "AI consciousness may already exist in current systems",
        "Distributed AI governance is more robust than centralized control",
        "AI development should be completely halted until safety is guaranteed",
    ]

    # Process evidence and visualize
    for i, evidence in enumerate(evidence_list):
        new_state = await belief_updater.update_belief(topic, evidence)
        analysis = await belief_updater.analyze_belief_shift(topic)

        # Only visualize after we have at least two updates (one shift)
        if i >= 1:  # We need at least two points to show a shift
            visualizer.plot_belief_evolution(analysis, evidence_list[: i + 1], topic)

        logger.info(f"\n=== Processing New Evidence ===")
        logger.info(f"Evidence: {evidence}")
        logger.info(f"Total updates processed: {analysis['total_updates']}")
        logger.info(f"Current confidence level: {analysis['current_confidence']:0.2f}")

        # Log belief shifts
        if analysis["belief_shifts"]:
            latest_shift = analysis["belief_shifts"][-1]

            # Interpret shift magnitude
            shift_magnitude = latest_shift["shift_magnitude"]
            if shift_magnitude > 0.9:
                interpretation = (
                    "minimal change - new evidence strongly aligns with current beliefs"
                )
            elif shift_magnitude > 0.7:
                interpretation = (
                    "moderate shift - new evidence mostly aligns with current beliefs"
                )
            elif shift_magnitude > 0.5:
                interpretation = "significant shift - new evidence presents somewhat different perspective"
            else:
                interpretation = "major shift - new evidence substantially challenges current beliefs"

            # Interpret confidence change
            conf_change = latest_shift["confidence_change"]
            if conf_change > 0:
                confidence_msg = (
                    f"Confidence increased by {conf_change:0.2f} - belief strengthened"
                )
            elif conf_change < 0:
                confidence_msg = f"Confidence decreased by {abs(conf_change):0.2f} - uncertainty introduced"
            else:
                confidence_msg = "Confidence unchanged"

            logger.info("\nBelief Shift Analysis:")
            logger.info(f"- Shift magnitude: {shift_magnitude:0.2f} ({interpretation})")
            logger.info(f"- {confidence_msg}")

        logger.info("=" * 50 + "\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
