import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


"""
Historical Context and Implementation Notes
----------------------------------------
This code implements Bayesian inference for coin flips, a classical problem that dates
back to Thomas Bayes (1701-1761) and Laplace's work on probability theory.

Key Historical Points:
- The Beta distribution used here was first studied by Euler and later applied to
  Bayesian statistics by Harold Jeffreys in the 1930s
- The Beta-Binomial conjugate prior relationship makes this an elegant example of
  Bayesian updating
- This implementation uses the modern approach of sequential updating, which wasn't
  computationally feasible in early Bayesian statistics

Technical Implementation:
- Uses Beta(α,β) as conjugate prior for Bernoulli likelihood
- α represents accumulated heads + prior
- β represents accumulated tails + prior
- Confidence intervals use quantiles of Beta distribution
"""

class BayesianCoinFlip:
    def __init__(self, alpha=1, beta=1):
        """
        Initialize with Beta(1,1) prior - represents uniform prior belief
        This is known as "Bayes-Laplace prior" or "flat prior"
        """
        self.alpha = alpha
        self.beta = beta
        self.history = []

    def update(self, data):
        """
        Conjugate update rule:
        - New α = old α + observed heads
        - New β = old β + observed tails
        This is mathematically equivalent to full Bayes' theorem but more efficient
        """
        heads = sum(data)
        tails = len(data) - heads
        self.alpha += heads
        self.beta += tails
        self.history.append(self.alpha / (self.alpha + self.beta))

    def get_confidence_interval(self):
        return stats.beta.interval(0.95, self.alpha, self.beta)


"""
Experimental Design Notes:
-------------------------
The experiment compares different batch sizes to demonstrate:
1. Speed of convergence
2. Uncertainty reduction
3. Trade-off between update frequency and computational efficiency

The true probability is set very high (0.999) to demonstrate:
- How quickly different batch sizes can detect extreme probabilities
- The effect of batch size on confidence interval width
- The balance between exploration and exploitation in learning
"""

# Let's compare different learning speeds!
np.random.seed(42)  # Keep it reproducible

# Create three coins with different batch sizes
small_batch = BayesianCoinFlip()  # 5 flips per update
medium_batch = BayesianCoinFlip()  # 20 flips per update
large_batch = BayesianCoinFlip()  # 50 flips per update

true_prob = 0.999
total_flips = 200

# Process different batch sizes
batch_sizes = {"Small Batch (5)": 5, "Medium Batch (20)": 20, "Large Batch (50)": 50}

results = {}
for name, coin in [
    ("Small Batch (5)", small_batch),
    ("Medium Batch (20)", medium_batch),
    ("Large Batch (50)", large_batch),
]:
    batch_size = batch_sizes[name]
    for _ in range(total_flips // batch_size):
        flips = np.random.random(batch_size) < true_prob
        coin.update(flips)
    results[name] = coin.history

# Plot the learning curves
plt.figure(figsize=(15, 6))

# Learning curves
plt.subplot(1, 2, 1)
for name, history in results.items():
    plt.plot(np.linspace(0, total_flips, len(history)), history, label=name)
plt.axhline(y=true_prob, color="r", linestyle="--", label="True Probability")
plt.title("Learning Curves: Effect of Batch Size")
plt.xlabel("Number of Coin Flips")
plt.ylabel("Estimated P(Heads)")
plt.legend()
plt.grid(True)

# Final beliefs
plt.subplot(1, 2, 2)
x = np.linspace(0, 1, 100)
for name, coin in [
    ("Small Batch (5)", small_batch),
    ("Medium Batch (20)", medium_batch),
    ("Large Batch (50)", large_batch),
]:
    y = stats.beta.pdf(x, coin.alpha, coin.beta)
    plt.plot(x, y, label=name)
    # Get confidence interval
    ci = coin.get_confidence_interval()
    print(f"{name} 95% Confidence Interval: {ci[0]:.3f} to {ci[1]:.3f}")

plt.axvline(x=true_prob, color="r", linestyle="--", label="True Probability")
plt.title("Final Beliefs After 200 Flips")
plt.xlabel("Probability of Heads")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
