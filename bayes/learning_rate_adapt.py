import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


"""
Bayesian Learning Rate Adaptation
--------------------------------
This implementation uses Bayesian inference to dynamically adjust neural network 
learning rates based on training success. It demonstrates how Bayesian methods
can be applied to optimization problems.

Key Concepts:
------------
1. Success-Based Adaptation: Learning rate adjusts based on improvement frequency
2. Beta Distribution: Models uncertainty in success probability
3. Confidence-Driven Updates: Higher confidence leads to smaller learning rates
4. Batch-wise Processing: Updates occur after processing groups of training steps

Mathematical Foundation:
----------------------
- Uses Beta(α,β) distribution to model success probability
- α represents accumulated successes (loss improvements)
- β represents accumulated failures (loss degradations)
- Confidence = α/(α+β) represents certainty in current performance
"""

class BayesianLearningRateAdapter:
    def __init__(self, initial_lr=0.01, alpha=1, beta=1):
        """
        Initialize with Beta(1,1) prior - uniform prior belief about success rate
        
        Parameters:
        - initial_lr: Base learning rate (0.01 is a common deep learning default)
        - alpha: Prior successes (default=1 for uniform prior)
        - beta: Prior failures (default=1 for uniform prior)
        """
        self.alpha = alpha
        self.beta = beta
        self.base_lr = initial_lr
        self.lr_history = []
        self.confidence_history = []

    def update_from_batch(self, loss_improved, batch_size):
        """
        Bayesian update based on batch performance
        
        Key Features:
        1. Conjugate Update: Updates Beta distribution parameters directly
        2. Inverse Scaling: Learning rate decreases with increasing confidence
        3. Floor Protection: Prevents learning rate from reaching zero
        
        The learning rate adaptation follows:
        lr = base_lr * (1 - confidence + floor)
        where:
        - confidence = α/(α+β)
        - floor = 0.1 (prevents complete stopping)
        """
        successes = sum(loss_improved)
        failures = batch_size - successes

        # Conjugate update for Beta distribution
        self.alpha += successes
        self.beta += failures

        # Calculate confidence as posterior mean
        confidence = self.alpha / (self.alpha + self.beta)
        self.confidence_history.append(confidence)

        # Inverse confidence scaling with floor
        new_lr = self.base_lr * (1 - confidence + 0.1)
        self.lr_history.append(new_lr)

        return new_lr

    def get_confidence_interval(self):
        """
        Calculates 95% credible interval for success probability
        Uses Beta distribution quantiles
        """
        return stats.beta.interval(0.95, self.alpha, self.beta)


"""
Simulation Design:
-----------------
The simulation demonstrates learning rate adaptation with:
1. Improving Success Rate: Models training progress
2. Batch Processing: Updates based on groups of training steps
3. Visualization: Shows both learning rate and confidence evolution

Key Parameters:
- total_steps = 100: Length of simulation
- batch_size = 32: Common deep learning batch size
- p_success: Increases from 0.2 to 0.8 (modeling improving model performance)

The simulation reveals:
1. Learning rate decreases as confidence increases
2. Adaptation to improving performance
3. Smooth transitions in both learning rate and confidence
"""

# Visualization Strategy:
# ---------------------
# Two-panel approach:
# 1. Learning Rate Panel: Shows adaptation over time
# 2. Confidence Panel: Shows increasing certainty in performance
#
# This layout enables direct comparison between:
# - How learning rate responds to confidence changes
# - How confidence builds up during successful training

# Simulate with increasing success rate (like your coin getting better)
adapter = BayesianLearningRateAdapter(initial_lr=0.01)
total_steps = 100
batch_size = 32

for step in range(total_steps):
    # Now we get BETTER over time (more successes)
    p_success = 0.2 + (0.6 * step / total_steps)  # Starts at 0.2, goes up to 0.8
    batch_results = np.random.random(batch_size) < p_success

    new_lr = adapter.update_from_batch(batch_results, batch_size)

# Visualization (using your plotting style!)
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(adapter.lr_history, label="Learning Rate")
plt.title("Adaptive Learning Rate Over Time")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(adapter.confidence_history, label="Model Confidence")
plt.title("Model Confidence Over Time")
plt.xlabel("Training Steps")
plt.ylabel("Confidence Score")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
