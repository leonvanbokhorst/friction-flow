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
    - belief_vector: High-dimensional vector representing semantic meaning
    - confidence: Scalar value [0-1] indicating certainty in current belief
    - prior_states: Historical record of previous beliefs and confidences
    - themes: List of identified themes (currently unused but prepared for future)
    """

    belief_vector: np.ndarray
    confidence: float
    prior_states: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)


class BayesianBeliefUpdater:
    """
    Core class that implements Bayesian belief updating using language model embeddings.
    Key concepts:
    1. Each topic maintains its own belief state
    2. Updates are performed using Bayesian inference
    3. Beliefs are represented as high-dimensional embeddings
    4. Confidence is updated based on similarity between current beliefs and new evidence
    """

    def __init__(self, llm: LanguageModel):
        self.llm = llm
        self.belief_states: Dict[str, BeliefState] = {}
        self.theme_weights = {}
        self.logger = logging.getLogger(__name__)

    async def initialize_belief_state(self, topic: str) -> BeliefState:
        """
        Initialize a new belief state for a given topic.
        """
        # Generate initial embedding for the topic
        embedding = await self.llm.generate_embedding(topic)

        # Create initial belief state
        belief_state = BeliefState(
            belief_vector=np.array(embedding),
            confidence=0.5,  # Start with moderate confidence
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

        # Store current state in history
        current_state.prior_states.append(
            (current_state.belief_vector.copy(), current_state.confidence)
        )

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
