import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class BayesianLearningRateAdapter:
    def __init__(self, initial_lr=0.01, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.base_lr = initial_lr
        self.lr_history = []
        self.confidence_history = []
        self.current_lr = initial_lr
        self.previous_loss = None

    def update_from_batch(self, current_loss):
        if self.previous_loss is None:
            self.previous_loss = current_loss
            self.confidence_history.append(0.5)
            self.lr_history.append(self.base_lr)
            return self.base_lr

        # Calculate relative improvement with noise threshold
        rel_improvement = (self.previous_loss - current_loss) / self.previous_loss
        
        # More nuanced success/failure signals
        if rel_improvement > 0.05:  # Significant improvement
            successes = 10
            failures = 0
        elif rel_improvement > 0.01:  # Good improvement
            successes = 5
            failures = 0
        elif rel_improvement > 0:  # Small improvement
            successes = 1
            failures = 0
        elif rel_improvement > -0.01:  # Small regression
            successes = 0
            failures = 1
        else:  # Significant regression
            successes = 0
            failures = 5

        # Adaptive momentum based on stability
        stability = 1.0 - abs(rel_improvement)
        momentum = min(0.98, max(0.90, stability))
        
        # Update Bayesian beliefs
        self.alpha = self.alpha * momentum + successes
        self.beta = self.beta * momentum + failures

        # Calculate confidence with bounds
        confidence = max(0.1, min(0.9, self.alpha / (self.alpha + self.beta)))
        self.confidence_history.append(confidence)

        # Adjust learning rate with bounds
        exploration_factor = 2.0 * (1.0 - confidence)
        new_lr = self.base_lr * max(0.1, min(2.0, exploration_factor))
        self.current_lr = new_lr
        self.lr_history.append(new_lr)

        self.previous_loss = current_loss
        return new_lr


class BayesianMNISTTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Initialize model (simple CNN for MNIST)
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(device)
        
        # Initialize optimizer and learning rate adapter
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.lr_adapter = BayesianLearningRateAdapter()
        
        # Training history
        self.loss_history = []

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        epoch_losses = []
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            # Training step
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Update learning rate based on loss
            new_lr = self.lr_adapter.update_from_batch(loss.item())

            # Update optimizer with new learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

            self.optimizer.step()

            # Smooth loss for plotting
            running_loss = 0.95 * running_loss + 0.05 * loss.item()
            epoch_losses.append(running_loss)
            self.loss_history.append(running_loss)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}]\t"
                    f"Loss: {running_loss:.6f}\t"
                    f"LR: {new_lr:.6f}\t"
                    f"Confidence: {self.lr_adapter.confidence_history[-1]:.3f}"
                )

        return np.mean(epoch_losses)


def plot_training_progress(trainer, fig=None, step=None):
    if fig is None:
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        ((ax1, ax2), (ax3, ax4)) = fig.subplots

    # Learning Rate vs Confidence scatter plot
    ax1.clear()
    ax1.scatter(trainer.lr_adapter.confidence_history, 
                trainer.lr_adapter.lr_history, 
                alpha=0.5, s=1)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate vs Confidence')
    ax1.grid(True)

    # Smoothed loss plot
    ax2.clear()
    window = 50
    smoothed_loss = np.convolve(trainer.loss_history, 
                               np.ones(window)/window, 
                               mode='valid')
    ax2.plot(smoothed_loss, label='Smoothed Loss')
    ax2.set_yscale('log')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Loss Progress')
    ax2.grid(True)
    ax2.legend()

    # Learning Rate plot
    ax3.clear()
    ax3.plot(trainer.lr_adapter.lr_history)
    ax3.set_title("Learning Rate")
    ax3.set_xlabel("Steps")

    # Model Confidence plot
    ax4.clear()
    ax4.plot(trainer.lr_adapter.confidence_history)
    ax4.set_title("Model Confidence")
    ax4.set_xlabel("Steps")

    return fig


def main():
    # Set up MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=32,
        shuffle=True,
    )

    trainer = BayesianMNISTTrainer()
    
    try:
        for epoch in range(5):
            loss = trainer.train_epoch(train_loader, epoch)
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Only show plot at the end
    plot_training_progress(trainer)
    plt.show()


if __name__ == "__main__":
    main()
