import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer  # For real text processing
import seaborn as sns
from collections import deque
import numpy as np


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch
import torch.nn as nn


class BrainInABoxV4(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        dropout=0.1,
        memory_size=100,
        num_layers=2,
        num_heads=8,
        activation="relu",
    ):
        super().__init__()

        # Enhanced configuration
        self.config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "memory_size": memory_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "activation": activation,
        }

        # Core architecture with configurable activation
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation=activation,
        )
        self.reasoning = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Enhanced state representation
        self.state_repr = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # Additional layer for deeper processing
            nn.LayerNorm(hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
        )

        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # Enhanced memory tracking
        self.hidden_dim = hidden_dim
        self.memory_states = deque(maxlen=memory_size)
        self.attention_patterns = deque(maxlen=memory_size)
        self.attention_maps = []

        # Enhanced metrics tracking
        self.metrics = {
            "memory_evolution": [],
            "attention_dynamics": [],
            "state_transitions": [],
            "gradient_norms": [],
            "layer_activations": [],
            "confidence_scores": [],
        }

        # New: Adaptive learning components
        self.adaptive_threshold = nn.Parameter(torch.tensor([0.5]))
        self.confidence_weights = nn.Parameter(torch.ones(num_heads))

        # New: State analysis tools
        self.state_analyzer = StateAnalyzer(hidden_dim)

    def _get_activation(self, activation_name):
        """Dynamic activation function selection"""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation_name.lower(), nn.ReLU())

    def forward(self, x, state=None, return_attention=False):
        # Enhanced state initialization
        if state is None:
            state = self.initialize_state(x)

        # Enhanced attention capture
        attention_weights = []

        def hook_fn(module, input, output):
            # Handle different output formats safely
            if isinstance(output, tuple):
                # Some implementations return (output, attention_weights)
                if len(output) > 1 and output[1] is not None:
                    attention_weights.append(output[1].detach())
            else:
                # If output is just the tensor, we'll use a different approach
                attention_weights.append(output.detach())

        # Register hooks for attention capture
        handles = []
        for layer in self.reasoning.layers:
            handles.append(layer.self_attn.register_forward_hook(hook_fn))

        try:
            # Enhanced forward pass with confidence scoring
            emb = self.embedding(x)
            reasoned = self.reasoning(emb)
            new_state = self.state_repr(reasoned[:, -1, :])

            # Adaptive state update
            confidence_score = torch.sigmoid(
                torch.matmul(new_state, state.transpose(-2, -1)).mean()
            )
            new_state = confidence_score * new_state + (1 - confidence_score) * state

            output = self.output_projection(reasoned)

            # Enhanced monitoring - only if we captured attention weights
            if attention_weights:
                self._update_monitoring_data(
                    attention_weights, new_state, confidence_score
                )

            if return_attention:
                return output, new_state, attention_weights
            return output, new_state

        finally:
            # Always clean up hooks
            for handle in handles:
                handle.remove()

    def _update_monitoring_data(self, attention_weights, new_state, confidence_score):
        """Enhanced monitoring data updates"""
        # Existing monitoring
        self.memory_states.append(new_state.mean(dim=0).detach().cpu())

        # Enhanced attention processing
        if attention_weights:
            processed_attention = self._process_attention_weights(attention_weights)
            self.attention_patterns.append(processed_attention)
            self.attention_maps.append(processed_attention)

        # Enhanced metrics
        self._update_metrics(new_state, confidence_score)

    def _process_attention_weights(self, attention_weights):
        """Process and analyze attention patterns"""
        # Combine attention from all layers
        combined_attention = torch.stack(attention_weights)
        # Weight by learned confidence
        weighted_attention = combined_attention * self.confidence_weights.view(-1, 1, 1)
        return weighted_attention.mean(dim=0).cpu()

    def _update_metrics(self, new_state, confidence_score):
        """Update enhanced metrics"""
        if len(self.memory_states) > 1:
            self.metrics["memory_evolution"].append(
                torch.norm(self.memory_states[-1] - self.memory_states[-2]).item()
            )
            self.metrics["confidence_scores"].append(confidence_score.item())

            # Add gradient tracking if training
            if self.training and new_state.grad is not None:
                self.metrics["gradient_norms"].append(torch.norm(new_state.grad).item())

    def initialize_state(self, x):
        """Enhanced state initialization"""
        batch_size = x.size(0)
        device = x.device

        # Initialize with learned parameters
        init_state = torch.randn(batch_size, self.hidden_dim, device=device)
        init_state = init_state * self.adaptive_threshold
        return init_state

    def visualize_brain_activity(self, show_memory=True, show_attention=True):
        """Comprehensive visualization of brain activity"""
        if not (self.memory_states or self.attention_patterns):
            print("No monitoring data available yet")
            return

        plt.figure(figsize=(15, 10))

        if show_memory and self.memory_states:
            plt.subplot(2, 1, 1)
            states = torch.stack(list(self.memory_states))
            sns.heatmap(states.numpy(), cmap="RdYlBu_r")
            plt.title("Memory Evolution")
            plt.ylabel("Time Step")
            plt.xlabel("Memory Dimension")

        if show_attention and self.attention_patterns:
            plt.subplot(2, 1, 2)
            attention_data = torch.stack(list(self.attention_patterns))
            sns.heatmap(attention_data.mean(0).numpy(), cmap="viridis")
            plt.title("Average Attention Pattern")
            plt.xlabel("Token Position")
            plt.ylabel("Token Position")

        plt.tight_layout()
        plt.show()


class StateAnalyzer:
    """New component for analyzing brain states"""

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.state_history = []

    def analyze_state(self, state):
        """Analyze state characteristics"""
        self.state_history.append(state.detach())

        analysis = {
            "complexity": self._compute_complexity(state),
            "stability": self._compute_stability(),
            "patterns": self._detect_patterns(),
        }
        return analysis

    def _compute_complexity(self, state):
        return torch.norm(state, p="fro").item()

    def _compute_stability(self):
        if len(self.state_history) < 2:
            return None
        return torch.norm(self.state_history[-1] - self.state_history[-2]).item()

    def _detect_patterns(self):
        if len(self.state_history) < 3:
            return None
        # Implement pattern detection logic
        return None


# Let's take it for a spin!🚗
def test_drive_brain():
    # Create a small model for testing
    model = BrainInABoxV4(
        vocab_size=1000,  # Smaller for testing
        embed_dim=64,  # Compact but functional
        hidden_dim=128,  # Decent memory size
    )

    # Create some fake input data
    batch_size = 3
    seq_length = 5
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))

    # Run it!
    reasoned_output, new_state = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Reasoning output shape: {reasoned_output.shape}")
    print(f"New state shape: {new_state.shape}")

    return model


if __name__ == "__main__":
    model = test_drive_brain()


# Create a simple training loop with visualization
def train_and_visualize(model, epochs=5, batch_size=32):
    # Generate some dummy sequential data
    seq_length = 10
    vocab_size = 1000
    dataset_size = 1000

    # Create synthetic data
    X = torch.randint(0, vocab_size, (dataset_size, seq_length))
    # Create targets as next token prediction
    y = torch.randint(0, vocab_size, (dataset_size,))

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []

    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            # Forward pass
            output, state = model(batch_x)

            # Get predictions for the last token
            logits = output[:, -1, :]  # [batch_size, vocab_size]

            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Visualize attention patterns periodically
        if epoch % 2 == 0 and hasattr(model, "attention_maps") and model.attention_maps:
            visualize_attention(model.attention_maps[-1], epoch)

    return losses


def visualize_attention(attention_map, epoch):
    plt.figure(figsize=(8, 6))

    try:
        # Convert to numpy and ensure 2D
        attention_display = attention_map.cpu().numpy()

        # If 1D, reshape to 2D square
        if len(attention_display.shape) == 1:
            size = int(np.sqrt(len(attention_display)))
            attention_display = attention_display.reshape(size, size)

        # Create heatmap
        sns.heatmap(attention_display, cmap="viridis", center=0)
        plt.title(f"Attention Pattern - Epoch {epoch+1}")
        plt.xlabel("Token Position")
        plt.ylabel("Token Position")
        plt.show()
    except Exception as e:
        print(f"Warning: Could not visualize attention map: {e}")
        plt.close()


# Let's run it!
vocab_size = 1000
model = BrainInABoxV4(
    vocab_size=vocab_size,  # Use same vocab_size as in training data
    embed_dim=64,
    hidden_dim=128,
)
losses = train_and_visualize(model)

# Plot training progress
plt.figure(figsize=(10, 5))
plt.plot(losses, "b-", label="Training Loss")
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


class TextBrainAnalyzer:
    def __init__(self, model_config=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = (
            BrainInABoxV4(vocab_size=50257, embed_dim=256, hidden_dim=512)
            if model_config is None
            else model_config
        )

        # Initialize pattern storage with empty lists
        self.pattern_store = {
            "narrative": {"memory": [], "attention": []},
            "analytical": {"memory": [], "attention": []},
        }

        # Generate more varied sample texts
        self.default_texts = {
            "narrative": [
                "One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin.",
                "The old man and the sea was a tale of courage and perseverance, as the fisherman battled both nature and his own limitations.",
                "In the quiet village, beneath the ancient oak trees, stories were passed down through generations.",
                "The detective examined the crime scene carefully, noting every detail that might lead to solving the mystery.",
                "Through the mist, she could barely make out the outline of the ancient castle on the hill.",
            ],
            "analytical": [
                "The analysis of artificial intelligence systems reveals complex patterns of information processing.",
                "Quantum mechanics demonstrates the probabilistic nature of subatomic particles and their interactions.",
                "Economic systems exhibit emergent properties through the collective behavior of individual agents.",
                "The fundamental principles of thermodynamics govern energy transfer in closed systems.",
                "Statistical analysis of large datasets requires careful consideration of sampling methodologies.",
            ],
        }

        self.text_categories = {
            "narrative": {
                "fiction": "stories, novels, creative writing",
                "personal": "blogs, diaries, memoirs",
                "journalistic": "news articles, features",
            },
            "analytical": {
                "scientific": "research papers, technical docs",
                "philosophical": "theoretical discussions",
                "business": "reports, analysis documents",
            },
            "conversational": {
                "dialogue": "transcripts, chat logs",
                "social": "social media posts",
                "informal": "casual communications",
            },
        }

        # Add data quality metrics
        self.data_metrics = {
            "samples_per_category": {},
            "avg_length": {},
            "vocabulary_diversity": {},
            "complexity_scores": {},
        }

    def process_text(self, text_path=None, text_type="narrative"):
        """Process multiple text samples while tracking memory evolution"""
        try:
            if text_path:
                try:
                    texts = [Path(text_path).read_text()]
                except FileNotFoundError:
                    print(
                        f"File {text_path} not found, using default {text_type} texts..."
                    )
                    texts = self.default_texts[text_type]
            else:
                print(f"Using default {text_type} texts...")
                texts = self.default_texts[text_type]

            print(f"Processing {len(texts)} {text_type} texts...")

            # Process each text sample
            for idx, text in enumerate(texts, 1):
                tokens = self.tokenizer.encode(text)
                input_tensor = torch.tensor([tokens])

                output, state = self.model(input_tensor)

                if output is not None and state is not None:
                    # Store memory patterns
                    self.pattern_store[text_type]["memory"].append(
                        state.mean(dim=0).detach().cpu()
                    )

                    # Store attention patterns if available
                    if (
                        hasattr(self.model, "attention_patterns")
                        and self.model.attention_patterns
                    ):
                        self.pattern_store[text_type]["attention"].extend(
                            [
                                p.mean().detach().cpu()
                                for p in self.model.attention_patterns
                            ]
                        )

                print(f"Processed sample {idx}/{len(texts)}")

            print(
                f"Stored patterns for {text_type}: Memory={len(self.pattern_store[text_type]['memory'])}, Attention={len(self.pattern_store[text_type]['attention'])}"
            )
            return True

        except Exception as e:
            print(f"Error processing text: {e}")
            return False

    def compare_text_types(self):
        """Compare patterns between narrative and analytical texts"""
        if not all(
            len(self.pattern_store[t]["memory"]) > 0
            for t in ["narrative", "analytical"]
        ):
            print("Error: Not enough data collected for comparison")
            return

        plt.figure(figsize=(15, 10))

        # Plot memory patterns
        plt.subplot(2, 1, 1)
        for text_type in ["narrative", "analytical"]:
            if self.pattern_store[text_type]["memory"]:
                patterns = torch.stack(self.pattern_store[text_type]["memory"])
                mean_pattern = patterns.mean(dim=0).numpy()
                std_pattern = patterns.std(dim=0).numpy()
                x = np.arange(len(mean_pattern))

                plt.plot(x, mean_pattern, label=f"{text_type.capitalize()} (mean)")
                plt.fill_between(
                    x, mean_pattern - std_pattern, mean_pattern + std_pattern, alpha=0.2
                )

        plt.title("Memory Pattern Comparison")
        plt.xlabel("Memory Dimension")
        plt.ylabel("Activation")
        plt.legend()

        # Plot attention patterns
        plt.subplot(2, 1, 2)
        for text_type in ["narrative", "analytical"]:
            if self.pattern_store[text_type]["attention"]:
                attention_patterns = torch.stack(
                    [
                        torch.tensor(p)
                        for p in self.pattern_store[text_type]["attention"]
                    ]
                )
                mean_attention = attention_patterns.mean(dim=0).numpy()
                std_attention = attention_patterns.std(dim=0).numpy()
                x = np.arange(len(mean_attention))

                plt.plot(x, mean_attention, label=f"{text_type.capitalize()} (mean)")
                plt.fill_between(
                    x,
                    mean_attention - std_attention,
                    mean_attention + std_attention,
                    alpha=0.2,
                )

        plt.title("Attention Pattern Comparison")
        plt.xlabel("Token Position")
        plt.ylabel("Attention Strength")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def analyze_complexity(self):
        """Analyze text complexity through pattern variability"""
        if not all(
            len(self.pattern_store[t]["memory"]) > 0
            for t in ["narrative", "analytical"]
        ):
            print("Error: Not enough data collected for complexity analysis")
            return

        plt.figure(figsize=(10, 6))

        for text_type in ["narrative", "analytical"]:
            if self.pattern_store[text_type]["memory"]:
                patterns = torch.stack(self.pattern_store[text_type]["memory"])
                variability = patterns.std(dim=0).numpy()
                plt.plot(variability, label=f"{text_type.capitalize()}")

        plt.title("Pattern Complexity Analysis")
        plt.xlabel("Memory Dimension")
        plt.ylabel("Pattern Variability")
        plt.legend()
        plt.show()

    def analyze_patterns(self):
        """Advanced pattern analysis"""
        results = {
            "temporal_patterns": self._analyze_temporal_patterns(),
            "cross_category_correlations": self._analyze_correlations(),
            "complexity_metrics": self._calculate_complexity_metrics(),
            "attention_dynamics": self._analyze_attention_flow(),
        }

        # Visualization enhancements
        self._plot_advanced_metrics(results)
        return results

    def _analyze_temporal_patterns(self):
        """Analyze how patterns evolve over time"""
        temporal_features = {
            "memory_evolution": [],
            "attention_shifts": [],
            "state_transitions": [],
        }
        # Implementation here
        return temporal_features


# Example usage
analyzer = TextBrainAnalyzer()

# Process both text types with multiple samples
print("\nProcessing narrative texts...")
analyzer.process_text(text_type="narrative")

print("\nProcessing analytical texts...")
analyzer.process_text(text_type="analytical")

# Compare patterns
print("\nComparing text patterns...")
analyzer.compare_text_types()

# Analyze complexity
print("\nAnalyzing pattern complexity...")
analyzer.analyze_complexity()


class ModelValidator:
    def __init__(self):
        self.validation_metrics = {
            "cross_validation_scores": [],
            "robustness_tests": [],
            "bias_metrics": [],
        }

    def validate_patterns(self, pattern_data):
        """Validate pattern recognition accuracy"""
        # Implementation here
        pass

    def test_generalization(self, test_data):
        """Test model generalization capabilities"""
        # Implementation here
        pass


class PatternVisualizer:
    def __init__(self):
        self.plot_config = {
            "style": "seaborn-darkgrid",
            "dimensions": (15, 10),
            "interactive": True,
        }

    def create_interactive_visualization(self, pattern_data):
        """Create interactive visualizations using plotly"""
        # Implementation here
        pass

    def generate_pattern_comparison(self, categories):
        """Generate comparative visualizations across categories"""
        # Implementation here
        pass


class DataIntegrator:
    def __init__(self):
        self.data_sources = {
            "academic": ["arxiv", "pubmed", "google_scholar"],
            "social": ["twitter", "reddit", "blog_feeds"],
            "professional": ["technical_docs", "business_reports"],
        }

    def fetch_and_process_data(self, source_type, parameters):
        """Fetch and process data from external sources"""
        # Implementation here
        pass


# Initialize model
model = BrainInABoxV4(
    vocab_size=50257,  # GPT-2 vocab size
    embed_dim=256,
    hidden_dim=512,
    memory_size=100,  # Number of states to track
)

# After training/inference
model.visualize_brain_activity()
