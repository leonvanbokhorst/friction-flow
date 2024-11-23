import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm  # For a nice progress bar
from coin_flipper import BayesianCoinFlip


"""
Batch Size Optimization Experiment
--------------------------------
This code explores the impact of batch size on Bayesian learning performance,
investigating the trade-offs between:
1. Learning accuracy (final and average error)
2. Convergence smoothness
3. Computational efficiency

Key Concepts:
------------
- Batch Learning: Processing multiple observations at once instead of one at a time
- Error Metrics: Both point-wise accuracy and learning trajectory smoothness
- Monte Carlo Simulation: Multiple runs to ensure robust results
- Confidence Measurement: Using posterior distribution width (1/(α+β))

Design Choices:
--------------
- Fixed total flips (200) ensures fair comparison across batch sizes
- Pre-generating all flips prevents sampling bias between experiments
- Multiple runs per batch size reduces random variation
- Various metrics capture different aspects of learning quality
"""

class BayesianBatchExplorer:
    def __init__(self, true_prob=0.7, total_flips=200):
        """
        Parameters:
        - true_prob: Actual probability of heads (0.7 chosen as non-trivial case)
        - total_flips: Fixed experiment length for fair comparison
        """
        self.true_prob = true_prob
        self.total_flips = total_flips

    def run_experiment(self, batch_size):
        """
        Conducts a single batch learning experiment.
        
        Metrics Captured:
        - mean_error: Average distance from truth (overall learning quality)
        - final_error: End-state accuracy (convergence quality)
        - confidence: Posterior width (uncertainty measurement)
        - convergence_speed: Sum of squared differences (learning smoothness)
        """
        coin = BayesianCoinFlip()
        errors = []  # Track distance from true probability

        # Generate all flips at once for fairness
        all_flips = np.random.random(self.total_flips) < self.true_prob

        for i in range(0, self.total_flips, batch_size):
            batch = all_flips[i : i + batch_size]
            coin.update(batch)
            current_estimate = coin.alpha / (coin.alpha + coin.beta)
            errors.append(abs(current_estimate - self.true_prob))

        return {
            "mean_error": np.mean(errors),
            "final_error": errors[-1],
            "confidence": 1 / (coin.alpha + coin.beta),  # Lower is more confident
            "convergence_speed": np.sum(np.diff(errors) ** 2),  # Lower is smoother
        }


# Experimental Setup and Analysis
# -----------------------------
# Batch sizes chosen to explore different scales:
# - Small (5-15): More updates, potentially noisy
# - Medium (20-30): Balance between updates and stability
# - Large (40-50): Fewer updates, more stable per update

# Results Processing
# -----------------
# Multiple runs per batch size to:
# 1. Reduce impact of random variations
# 2. Get more reliable performance metrics
# 3. Account for different possible learning trajectories

# Visualization Design
# -------------------
# Two-panel approach:
# 1. Error Analysis: Compares average vs final error
#    - Shows both learning quality and end-state performance
# 2. Convergence Analysis: Shows learning smoothness
#    - Lower values indicate more stable learning progression

# The experiment reveals:
# 1. Trade-off between update frequency and stability
# 2. Optimal batch sizes for different objectives
# 3. Relationship between batch size and learning characteristics

# Let's try a range of batch sizes
batch_sizes = [5, 10, 15, 20, 25, 30, 40, 50]
explorer = BayesianBatchExplorer()

results = {}
for batch_size in tqdm(batch_sizes):
    # Run each experiment multiple times for robustness
    batch_results = [explorer.run_experiment(batch_size) for _ in range(10)]

    # Average the results
    results[batch_size] = {
        "mean_error": np.mean([r["mean_error"] for r in batch_results]),
        "final_error": np.mean([r["final_error"] for r in batch_results]),
        "confidence": np.mean([r["confidence"] for r in batch_results]),
        "convergence_speed": np.mean([r["convergence_speed"] for r in batch_results]),
    }

# Plot the results
plt.figure(figsize=(15, 5))

# Error and Confidence
plt.subplot(1, 2, 1)
batch_sizes_array = np.array(batch_sizes)
plt.plot(
    batch_sizes_array,
    [results[b]["mean_error"] for b in batch_sizes],
    "b-o",
    label="Average Error",
)
plt.plot(
    batch_sizes_array,
    [results[b]["final_error"] for b in batch_sizes],
    "r-o",
    label="Final Error",
)
plt.xlabel("Batch Size")
plt.ylabel("Error")
plt.title("Error vs Batch Size")
plt.legend()
plt.grid(True)

# Convergence Speed
plt.subplot(1, 2, 2)
plt.plot(
    batch_sizes_array,
    [results[b]["convergence_speed"] for b in batch_sizes],
    "g-o",
    label="Convergence Speed",
)
plt.xlabel("Batch Size")
plt.ylabel("Convergence Measure (lower is smoother)")
plt.title("Learning Smoothness vs Batch Size")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Print the "best" batch size according to different metrics
best_final_error = min(results.items(), key=lambda x: x[1]["final_error"])
best_smooth = min(results.items(), key=lambda x: x[1]["convergence_speed"])

print(f"\nBest batch size for final accuracy: {best_final_error[0]}")
print(f"Best batch size for smooth learning: {best_smooth[0]}")
