"""Deep Belief Network (DBN) Demonstration on MNIST Dataset

This module implements a Deep Belief Network to learn hierarchical representations
of handwritten digits from the MNIST dataset. A DBN is a generative model composed
of multiple layers of Restricted Boltzmann Machines (RBMs) stacked together.

Key Concepts:
------------
1. Deep Belief Network (DBN):
   - A deep learning architecture that learns to probabilistically reconstruct its inputs
   - Composed of multiple RBM layers trained in a greedy layer-wise manner
   - Each layer learns increasingly abstract features of the data

2. Restricted Boltzmann Machine (RBM):
   - A two-layer neural network that learns to reconstruct input data
   - "Restricted" because there are no connections between nodes in the same layer
   - Uses contrastive divergence for training (positive and negative phases)

Experiment Overview:
------------------
This experiment:
1. Loads MNIST handwritten digit data (28x28 pixel images)
2. Creates a 3-layer DBN with dimensions: 784 -> 256 -> 64
   - 784: Input layer (28x28 flattened pixels)
   - 256: First hidden layer for low-level features
   - 64: Second hidden layer for higher-level abstractions

3. Generates comprehensive visualizations:
   - Input reconstructions: Shows how well the model recreates input images
   - Weight matrices: Visualizes learned features at each layer
   - Activation patterns: Shows how different inputs activate network nodes
   - Training metrics: Tracks reconstruction error over time

Training Process:
---------------
1. Layer-wise pretraining:
   - Each RBM layer is trained independently
   - Lower layers learn simple features (edges, corners)
   - Higher layers learn complex feature combinations

2. For each layer:
   - Forward pass: Compute hidden unit activations
   - Reconstruction: Generate visible unit reconstructions
   - Update weights using contrastive divergence
   - Track reconstruction error and visualize progress

Output Structure:
---------------
The experiment creates timestamped output directories containing:
- Reconstruction visualizations
- Weight matrix patterns
- Activation heatmaps
- Training metrics
- Configuration details

Usage:
-----
Run this script directly to train the DBN and generate visualizations:
    python dbn_mnist_demo.py

Requirements:
-----------
- NumPy: Numerical computations
- Matplotlib: Visualization
- Scikit-learn: MNIST dataset loading
- Seaborn: Enhanced visualizations
- tqdm: Progress tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Optional
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime


class RBM:
    """Enhanced Restricted Boltzmann Machine with visualization capabilities.
    
    An RBM is a two-layer neural network that learns to reconstruct input data through
    unsupervised learning. It consists of:
    - Visible layer: Represents the input data
    - Hidden layer: Learns features from the input
    - Weights: Bidirectional connections between layers
    
    The learning process involves:
    1. Positive phase: Computing hidden activations from input
    2. Negative phase: Reconstructing input from hidden activations
    3. Weight updates: Minimizing reconstruction error
    
    Args:
        n_visible (int): Number of visible units (input dimensions)
        n_hidden (int): Number of hidden units (learned features)
        learning_rate (float): Learning rate for weight updates
    """

    def __init__(self, n_visible: int, n_hidden: int, learning_rate: float = 0.01):
        self.weights = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)
        self.learning_rate = learning_rate
        self.training_losses = []

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

    def free_energy(self, v: np.ndarray) -> float:
        """Calculate the free energy of a visible vector."""
        wx_b = np.dot(v, self.weights) + self.hidden_bias
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)))
        vbias_term = np.dot(v, self.visible_bias)
        return -hidden_term - vbias_term

    def reconstruct(self, v: np.ndarray) -> np.ndarray:
        """Reconstruct visible units through one hidden layer and back."""
        h_prob, _ = self.sample_hidden(v)
        v_prob, _ = self.sample_visible(h_prob)
        return v_prob

    def sample_hidden(self, visible: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample hidden units given visible units."""
        hidden_probs = self.sigmoid(np.dot(visible, self.weights) + self.hidden_bias)
        hidden_states = (hidden_probs > np.random.random(hidden_probs.shape)).astype(
            float
        )
        return hidden_probs, hidden_states

    def sample_visible(self, hidden: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample visible units given hidden units."""
        visible_probs = self.sigmoid(np.dot(hidden, self.weights.T) + self.visible_bias)
        visible_states = (visible_probs > np.random.random(visible_probs.shape)).astype(
            float
        )
        return visible_probs, visible_states


class EnhancedDBN:
    """Enhanced Deep Belief Network with visualization and analysis capabilities.
    
    A DBN is created by stacking multiple RBMs, where each layer learns to represent
    features of increasing abstraction. This implementation includes:
    - Layer-wise pretraining
    - Comprehensive visualization tools
    - Progress tracking and metrics
    - Organized output management
    
    The network architecture is specified through layer_sizes, where:
    - First element is input dimension
    - Last element is final hidden layer size
    - Intermediate elements define hidden layer sizes
    
    Args:
        layer_sizes (List[int]): Dimensions of each layer
        learning_rate (float): Learning rate for all RBM layers
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """Initialize DBN with output directory creation."""
        self.rbm_layers = []
        self.layer_sizes = layer_sizes
        self.rbm_layers.extend(
            RBM(layer_sizes[i], layer_sizes[i + 1], learning_rate)
            for i in range(len(layer_sizes) - 1)
        )
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("output", f"dbn_run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create subdirectories for different visualization types
        self.viz_dirs = {
            "reconstructions": os.path.join(self.output_dir, "reconstructions"),
            "weights": os.path.join(self.output_dir, "weights"),
            "activations": os.path.join(self.output_dir, "activations"),
            "metrics": os.path.join(self.output_dir, "metrics"),
        }

        for dir_path in self.viz_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def pretrain(
        self,
        data: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        visualize: bool = True,
    ) -> None:
        """Enhanced pretraining with visualization and monitoring."""
        current_input = data

        for layer_idx, rbm in enumerate(self.rbm_layers):
            print(f"\nPretraining layer {layer_idx + 1}")

            for epoch in tqdm(range(epochs), desc=f"Layer {layer_idx + 1}"):
                reconstruction_errors = []

                # Mini-batch training with progress tracking
                for batch_start in range(0, len(data), batch_size):
                    batch = current_input[batch_start : batch_start + batch_size]
                    reconstruction_error = self._train_batch(rbm, batch)
                    reconstruction_errors.append(reconstruction_error)

                avg_error = np.mean(reconstruction_errors)
                rbm.training_losses.append(avg_error)

                if epoch % 2 == 0 and visualize:
                    self._visualize_training(rbm, layer_idx, epoch, batch)

            # Transform data for next layer
            current_input, _ = rbm.sample_hidden(current_input)

    def _train_batch(self, rbm: RBM, batch: np.ndarray) -> float:
        """Train RBM on a single batch and return reconstruction error."""
        # Positive phase
        pos_hidden_probs, pos_hidden_states = rbm.sample_hidden(batch)
        pos_associations = np.dot(batch.T, pos_hidden_probs)

        # Negative phase
        neg_visible_probs, _ = rbm.sample_visible(pos_hidden_states)
        neg_hidden_probs, _ = rbm.sample_hidden(neg_visible_probs)
        neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

        # Update weights and biases
        rbm.weights += rbm.learning_rate * (
            (pos_associations - neg_associations) / len(batch)
        )
        rbm.visible_bias += rbm.learning_rate * np.mean(
            batch - neg_visible_probs, axis=0
        )
        rbm.hidden_bias += rbm.learning_rate * np.mean(
            pos_hidden_probs - neg_hidden_probs, axis=0
        )

        return np.mean((batch - neg_visible_probs) ** 2)

    def _visualize_training(
        self, rbm: RBM, layer_idx: int, epoch: int, sample_batch: np.ndarray
    ) -> None:
        """Visualize training progress with multiple plots."""

        # Only show image reconstructions for the first layer
        if layer_idx == 0:
            plt.figure(figsize=(15, 5))

            # Plot 1: Sample reconstructions
            n_samples = 5
            samples = sample_batch[:n_samples]
            reconstructed = rbm.reconstruct(samples)

            for i in range(n_samples):
                plt.subplot(n_samples, 2, 2 * i + 1)
                plt.imshow(samples[i].reshape(28, 28), cmap="gray")
                plt.axis("off")
                if i == 0:
                    plt.title("Original")

                plt.subplot(n_samples, 2, 2 * i + 2)
                plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
                plt.axis("off")
                if i == 0:
                    plt.title("Reconstructed")

            self.save_visualization("reconstructions", layer_idx, epoch, "_reconstruction.png")
        # For all layers, show weight patterns
        plt.figure(figsize=(10, 10))
        n_hidden = min(100, rbm.weights.shape[1])
        # Only show weights as images for first layer
        if layer_idx == 0:
            n_grid = int(np.ceil(np.sqrt(n_hidden)))

            for i in range(n_hidden):
                plt.subplot(n_grid, n_grid, i + 1)
                plt.imshow(rbm.weights[:, i].reshape(28, 28), cmap="gray")
                plt.axis("off")
        else:
            # For higher layers, show weights as heatmaps
            plt.subplot(1, 1, 1)
            sns.heatmap(rbm.weights, cmap="viridis", center=0)
            plt.title(f"Layer {layer_idx + 1} Weight Matrix")

        plt.suptitle(f"Layer {layer_idx + 1} Features (Epoch {epoch})")
        self.save_visualization("weights", layer_idx, epoch, "_weights.png")
        # Add activation patterns visualization
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 1, 1)
        sns.heatmap(sample_batch[:10], cmap="viridis")
        plt.title(f"Layer {layer_idx + 1} Activation Patterns")
        self.save_visualization("activations", layer_idx, epoch, "_activations.png")
        # Save training metrics
        if hasattr(rbm, "training_losses"):
            plt.figure(figsize=(8, 4))
            plt.plot(rbm.training_losses)
            plt.title(f"Layer {layer_idx + 1} Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Reconstruction Error")
            self.save_visualization("metrics", layer_idx, epoch, "_training_loss.png")
            plt.close()

    def save_visualization(
        self, viz_type: str, layer_idx: int, epoch: int, file_suffix: str
    ) -> str:
        plt.tight_layout()
        result = os.path.join(
            self.viz_dirs[viz_type], f"layer{layer_idx}_epoch{epoch}{file_suffix}"
        )
        plt.savefig(result)
        plt.close()

        return result


def load_mnist(n_samples: int = 10000) -> np.ndarray:
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X[:n_samples]
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def analyze_representations(dbn: EnhancedDBN, data: np.ndarray) -> None:
    """Analyze and visualize the learned representations."""
    # Get activations for each layer
    activations = []
    current_input = data

    for rbm in dbn.rbm_layers:
        hidden_probs, _ = rbm.sample_hidden(current_input)
        activations.append(hidden_probs)
        current_input = hidden_probs

    # Visualize activation patterns
    plt.figure(figsize=(15, 5))
    for i, activation in enumerate(activations):
        plt.subplot(1, len(activations), i + 1)
        plt.title(f"Layer {i + 1} Activations")
        sns.heatmap(activation[:10].T, cmap="viridis")
    plt.tight_layout()
    plt.savefig("layer_activations.png")
    plt.close()


def main():
    """Run DBN training with organized output."""
    # Load and prepare MNIST data
    data = load_mnist()

    # Create and train DBN
    dbn = EnhancedDBN([784, 256, 64], learning_rate=0.01)

    # Save configuration
    config = {
        "layer_sizes": dbn.layer_sizes,
        "learning_rate": 0.01,
        "epochs": 10,
        "batch_size": 32,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(dbn.output_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # Train and generate visualizations
    dbn.pretrain(data, epochs=10, batch_size=32)

    # Analyze learned representations
    analyze_representations(dbn, data)


if __name__ == "__main__":
    main()
