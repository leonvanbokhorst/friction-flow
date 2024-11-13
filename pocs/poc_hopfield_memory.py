"""Hopfield Network Pattern Recognition and Memory Demonstration.

This module implements a Hopfield Network to demonstrate associative memory and pattern completion,
showing how neural networks can store and recover patterns even in the presence of noise.

Experiment Overview:
-------------------
The experiment demonstrates a simple pattern recognition system using a Hopfield Network
to store and retrieve 5x5 pixel representations of letters. It showcases three key
capabilities of associative memory:

1. Pattern Storage:
   - Two letter patterns ('H' and 'T') are stored in the network
   - Storage uses Hebbian learning to create a weight matrix
   - Each pattern becomes an attractor in the network's state space

2. Pattern Corruption:
   - The letter 'H' pattern is corrupted with random noise
   - 8 random pixels are flipped from their original state
   - This simulates real-world noise or partial information

3. Pattern Recovery:
   - The network processes the noisy pattern
   - Through iterative updates, it converges to the nearest stored pattern
   - Demonstrates the network's ability to perform error correction

Key Concepts Demonstrated:
-------------------------
- Associative Memory: Patterns are recovered by association, not address
- Attractor Dynamics: Network converges to stable states (stored patterns)
- Error Correction: Ability to clean up noisy or corrupted inputs
- Content-Addressable Memory: Retrieval based on partial or similar content

Technical Implementation:
-----------------------
- Uses binary threshold neurons (-1/1 states)
- Implements asynchronous updates
- Demonstrates both classical Hopfield dynamics and modern attention-like mechanisms
- Visualizes the process through matplotlib plots

The experiment shows how simple neural architectures can exhibit complex
cognitive-like behaviors such as pattern completion and error correction,
fundamental properties of biological memory systems.

Example Usage:
-------------
    python poc_hopfield_memory.py

This will run the demonstration and display three plots:
1. Original 'H' pattern
2. Noisy version of the pattern
3. Recovered pattern after network processing

Historical Context:
------------------
Hopfield Networks, introduced by John Hopfield in 1982, were among the first
neural architectures to demonstrate how distributed representations could serve
as content-addressable memory. Their principles influenced modern deep learning
and attention mechanisms.

References:
-----------
Hopfield, J. J. (1982). Neural networks and physical systems with emergent
collective computational abilities. Proceedings of the National Academy of
Sciences, 79(8), 2554-2558.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random


class HopfieldNetwork:
    """A Hopfield Network implementation for pattern storage and retrieval.

    A Hopfield network is a form of recurrent artificial neural network that serves as a content-addressable memory system.
    Key characteristics:
    - Symmetric weights (w_ij = w_ji)
    - Binary threshold neurons (-1 or 1)
    - No self-connections (diagonal weights = 0)
    - Asynchronous updates
    
    The network can:
    1. Store patterns through Hebbian learning
    2. Recover patterns from noisy or partial inputs
    3. Converge to stable states (attractors)

    Theoretical capacity (number of patterns) ≈ 0.15N, where N is network size.

    Based on Hopfield (1982) - Neural networks and physical systems with
    emergent collective computational abilities.
    """

    def __init__(self, size: int):
        """Initialize the Hopfield Network.

        Args:
            size: Number of neurons in the network
        """
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns: List[np.ndarray]) -> None:
        """Store patterns in the network using Hebbian learning.

        The training process:
        1. For each pattern, compute outer product of pattern with itself
        2. Sum these products to create weight matrix
        3. Zero diagonal to prevent self-connections
        4. Normalize by number of patterns

        Hebbian Rule: "Neurons that fire together, wire together"
        w_ij += x_i * x_j where x_i, x_j are neuron states

        Args:
            patterns: List of binary patterns to store (each element should be -1 or 1)
        """
        for pattern in patterns:
            # Hebbian learning rule: strengthen connections between co-active neurons
            pattern = pattern.reshape(-1, 1)
            self.weights += np.outer(pattern, pattern)

        # Zero out diagonal (no self-connections) and normalize
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def update(self, state: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Update network state until convergence or max iterations reached.

        The update process:
        1. Randomly select neurons for asynchronous update
        2. For each neuron:
           - Calculate local field (weighted sum of inputs)
           - Update state based on sign of local field
        3. Continue until convergence or max iterations

        Energy function: E = -1/2 ∑_ij w_ij s_i s_j
        Network always evolves toward local energy minima.

        Args:
            state: Initial state of the network (-1/1 values)
            max_iterations: Maximum number of iterations to run

        Returns:
            Final state of the network (a stored memory pattern or local minimum)
        """
        prev_state = state.copy()

        for _ in range(max_iterations):
            # Asynchronous update: update neurons in random order
            update_order = list(range(self.size))
            random.shuffle(update_order)

            for i in update_order:
                # Calculate local field
                h = np.dot(self.weights[i], state)
                # Update neuron state using sign activation function
                state[i] = 1 if h >= 0 else -1

            # Check for convergence
            if np.array_equal(state, prev_state):
                break
            prev_state = state.copy()

        return state

    def update_modern(self, state: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Modern continuous Hopfield update with attention-like mechanism.

        Args:
            state: Input state
            beta: Temperature parameter (controls softmax sharpness)
        """
        # Compute attention-like scores
        scores = np.exp(beta * np.dot(self.weights, state))
        # Softmax normalization
        return scores / np.sum(scores)


def create_letter_patterns() -> List[np.ndarray]:
    """Create binary patterns for letters 'H' and 'T'.
    
    Patterns are represented as 5x5 grids flattened to 25-element vectors.
    Values are binary (-1 for black, 1 for white).
    
    Returns:
        List containing two patterns: [H_pattern, T_pattern]
    """
    H = np.array(
        [
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
        ]
    )

    T = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
        ]
    )

    return [H, T]


def plot_pattern(pattern: np.ndarray, title: str) -> None:
    """Visualize a pattern as a 5x5 grid."""
    plt.imshow(pattern.reshape(5, 5), cmap="binary")
    plt.title(title)
    plt.axis("off")


def compare_mechanisms():
    """Compare traditional Hopfield dynamics with modern attention mechanisms.
    
    Traditional Hopfield:
    - Uses discrete states (-1/1)
    - Binary threshold activation
    - Energy minimization dynamics
    
    Modern Attention (Transformer-like):
    - Continuous states
    - Softmax activation
    - Query-Key-Value computation
    
    Both approaches implement associative memory through
    different mathematical frameworks.
    """


def main():
    """Run Hopfield Network demonstration.
    
    This experiment demonstrates:
    1. Pattern Storage: Training network on 'H' and 'T' patterns
    2. Pattern Completion: Recovering full pattern from noisy input
    3. Attractor Dynamics: Network converges to stored memory
    
    The visualization shows:
    - Original clean pattern
    - Noisy pattern (corrupted with random flips)
    - Recovered pattern after network convergence
    
    This illustrates the network's ability to perform:
    - Content-addressable memory
    - Pattern completion
    - Error correction
    """
    # Create and train network
    patterns = create_letter_patterns()
    network = HopfieldNetwork(25)
    network.train(patterns)

    # Create noisy version of 'H'
    noisy_pattern = patterns[0].copy()
    noise_positions = random.sample(range(25), 8)  # Add noise to 8 positions
    for pos in noise_positions:
        noisy_pattern[pos] *= -1

    # Recover pattern
    recovered_pattern = network.update(noisy_pattern.copy())

    # Visualize results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plot_pattern(patterns[0], "Original Pattern 'H'")

    plt.subplot(132)
    plot_pattern(noisy_pattern, "Noisy Pattern")

    plt.subplot(133)
    plot_pattern(recovered_pattern, "Recovered Pattern")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
