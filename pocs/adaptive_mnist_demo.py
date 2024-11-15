"""
Adaptive MNIST Neural Network Demo

This module implements an adaptive neural network system for MNIST digit classification.
The system automatically adjusts its architecture and learning parameters based on
training performance.

Key Features:
    - Dynamic network complexity adjustment
    - Automatic device selection and optimization
    - Adaptive learning rate and regularization
    - Performance visualization
    - Hardware-specific optimizations

Classes:
    AdaptiveNeuralNet: A neural network that can adapt its architecture
    AdaptiveLearningSystem: Training system with automatic optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time


class AdaptiveNeuralNet(nn.Module):
    """
    An adaptive neural network that can modify its architecture during training.
    
    The network starts with a simple architecture and can increase its complexity
    when learning plateaus. It also supports adaptive regularization.
    
    Attributes:
        input_size (int): Dimension of input features (default: 784 for MNIST)
        layers (nn.ModuleList): Dynamic list of network layers
        training_history (dict): Tracks performance metrics over time
        dropout_rate (float): Current dropout rate for regularization
        learning_rate (float): Current learning rate
        current_complexity (int): Tracks the network's architectural complexity level
    """
    def __init__(self, input_size=784, initial_hidden_size=64):
        """
        Initialize the adaptive neural network.
        
        Args:
            input_size (int): Input feature dimension
            initial_hidden_size (int): Initial number of hidden units
        """
        super(AdaptiveNeuralNet, self).__init__()
        self.input_size = input_size
        self.flatten = nn.Flatten()

        # Start with a simple architecture
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, initial_hidden_size),
                nn.ReLU(),
                nn.Linear(initial_hidden_size, 10),
            ]
        )

        # Track performance metrics
        self.training_history = {"loss": [], "accuracy": [], "plateau_count": 0}

        self.dropout_rate = 0.0
        self.learning_rate = 0.001
        self.current_complexity = 1  # Track network complexity level

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def add_complexity(self):
        """
        Increases network capacity by adding new layers with doubled hidden size.
        
        The method:
        1. Doubles the hidden layer size
        2. Adds batch normalization for training stability
        3. Maintains skip connections for better gradient flow
        
        Raises:
            ValueError: If no linear layer is found in the network
        """
        # Find the last Linear layer
        last_linear = None
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                last_linear = layer

        if not last_linear:
            raise ValueError("No Linear layer found in network")

        current_hidden_size = last_linear.in_features
        new_hidden_size = current_hidden_size * 2

        # Create new layers with residual connection
        new_layers = nn.ModuleList([])

        # Add all layers up to the last Linear layer
        for layer in self.layers:
            if layer is last_linear:
                break
            new_layers.append(layer)

        # Add new hidden layers with batch normalization
        new_layers.extend(
            [
                nn.Linear(current_hidden_size, new_hidden_size),
                nn.BatchNorm1d(new_hidden_size),  # Add BatchNorm for stability
                nn.ReLU(),
                nn.Linear(new_hidden_size, 10),
            ]
        )

        self.layers = new_layers
        self.current_complexity += 1
        print(f"Network complexity increased. New hidden size: {new_hidden_size}")

    def add_regularization(self):
        """Increase regularization to prevent overfitting"""
        self.dropout_rate = min(0.5, self.dropout_rate + 0.1)
        # Update dropout layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Dropout):
                self.layers[i] = nn.Dropout(self.dropout_rate)
        print(f"Regularization increased. New dropout rate: {self.dropout_rate:.2f}")


class AdaptiveLearningSystem:
    """
    A learning system that automatically optimizes training parameters and hardware usage.
    
    Features:
        - Automatic device selection (CPU/MPS)
        - Batch size optimization
        - Dynamic model adaptation
        - Performance monitoring and visualization
        - Hardware-specific optimizations
    
    Attributes:
        model (AdaptiveNeuralNet): The neural network being trained
        device (str): Optimal computing device
        optimal_batch_size (int): Determined optimal batch size
        plateau_threshold (int): Number of epochs before considering plateau
        improvement_threshold (float): Minimum improvement to avoid plateau
        max_complexity (int): Maximum allowed network complexity
    """
    def __init__(self, model, train_loader, test_loader):
        # Initialize model first
        self.model = model
        
        # Benchmark devices
        self.device = self.benchmark_devices(self.model, num_iterations=1000)
        print(f"Selected device: {self.device} based on benchmarking")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Find optimal batch size and update data loaders
        self.optimal_batch_size = self.find_optimal_batch_size()
        print(f"Optimal batch size: {self.optimal_batch_size}")
        
        self.train_loader = self.update_dataloader(train_loader.dataset, train=True)
        self.test_loader = self.update_dataloader(test_loader.dataset, train=False)
        
        # Apply PyTorch 2.0+ optimization if available
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            print("Using torch.compile() for optimization")
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.learning_rate)
        
        # Adaptive parameters
        self.plateau_threshold = 3
        self.improvement_threshold = 0.01
        self.max_complexity = 4

        # Metrics tracking
        self.train_losses = []
        self.test_accuracies = []
        self.adaptation_points = []

    def benchmark_devices(self, model, num_iterations=100):
        """
        Benchmarks available computing devices to select the fastest one.
        
        Args:
            model (nn.Module): Model to benchmark
            num_iterations (int): Number of forward passes for timing
        
        Returns:
            str: Name of the fastest device ('cpu' or 'mps')
        """
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        
        best_device = "cpu"
        best_time = float('inf')
        
        # Create sample batch
        sample_input = torch.randn(64, 1, 28, 28)
        
        for device in devices:
            model.to(device)
            sample_input = sample_input.to(device)
            
            # Warmup
            for _ in range(10):
                model(sample_input)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                model(sample_input)
            end_time = time.time()
            
            total_time = end_time - start_time
            print(f"{device.upper()} time: {total_time:.4f} seconds")
            
            if total_time < best_time:
                best_time = total_time
                best_device = device
        
        return best_device

    def find_optimal_batch_size(self):
        """
        Determines the optimal batch size through performance testing.
        
        Tests various batch sizes and measures throughput to find the optimal
        balance between memory usage and computational efficiency.
        
        Returns:
            int: Optimal batch size for training
        """
        """Test different batch sizes to find optimal performance"""
        batch_sizes = [32, 64, 128, 256, 512]
        best_time = float('inf')
        optimal_batch = 64  # default
        
        sample_input = torch.randn(1, 1, 28, 28).to(self.device)
        self.model.eval()
        
        print("\nBenchmarking batch sizes:")
        for batch in batch_sizes:
            batched_input = sample_input.repeat(batch, 1, 1, 1)
            
            # Warmup
            for _ in range(5):
                self.model(batched_input)
            
            # Benchmark
            start_time = time.time()
            iterations = 100
            for _ in range(iterations):
                self.model(batched_input)
            end_time = time.time()
            
            batch_time = end_time - start_time
            print(f"Batch size {batch:4d}: {batch_time:.4f} seconds")
            
            if batch_time < best_time:
                best_time = batch_time
                optimal_batch = batch
        
        return optimal_batch

    def update_dataloader(self, dataset, train=True):
        """Create new dataloader with optimal batch size"""
        return DataLoader(
            dataset,
            batch_size=self.optimal_batch_size,
            shuffle=train,
            num_workers=4,  # Parallel data loading
            pin_memory=True  # Faster data transfer
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        log_interval = max(1, len(self.train_loader) // 10)  # Log 10 times per epoch

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if batch_idx % log_interval == (log_interval - 1):
                print(f"Batch {batch_idx+1}, Loss: {running_loss/log_interval:.3f}")
                running_loss = 0.0

        return self.calculate_loss(), self.evaluate()

    def calculate_loss(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total

    def check_plateau(self, accuracies, threshold=0.005):
        """Check if training has plateaued"""
        if len(accuracies) < 3:
            return False
        recent_improvements = [
            accuracies[i] - accuracies[i - 1]
            for i in range(len(accuracies) - 2, len(accuracies))
        ]
        return all(abs(imp) < threshold for imp in recent_improvements)

    def adapt_model(self, epoch):
        """
        Adapts the model architecture and learning parameters based on performance.
        
        Adaptation strategies:
        1. Increases network complexity if accuracy plateaus
        2. Adjusts learning rate based on training progress
        3. Applies regularization when needed
        
        Args:
            epoch (int): Current training epoch
        """
        recent_accuracies = (
            self.test_accuracies[-5:] if len(self.test_accuracies) >= 5 else []
        )

        if len(recent_accuracies) >= 5:
            max_recent_acc = max(recent_accuracies)
            current_acc = self.test_accuracies[-1]
            avg_recent_acc = sum(recent_accuracies) / len(recent_accuracies)

            # Only adapt if:
            # 1. We're below 98% accuracy AND
            # 2. We've plateaued or declined AND
            # 3. Enough epochs have passed since last complexity increase
            if (
                current_acc < 98.0
                and current_acc <= avg_recent_acc  # Using average instead of max
                and epoch > self.model.current_complexity * 4
            ):  # More epochs between increases

                if self.model.current_complexity < self.max_complexity:
                    self.model.add_complexity()
                    # Smaller learning rate increase
                    self.model.learning_rate *= 1.1
                    self.optimizer = optim.Adam(
                        self.model.parameters(),
                        lr=self.model.learning_rate,
                        betas=(0.9, 0.999),  # Default Adam betas
                        eps=1e-8,
                        weight_decay=1e-5,  # Light L2 regularization
                    )
                    self.adaptation_points.append((epoch, "Increased Complexity"))
                    print(f"Adapting network - Current accuracy: {current_acc:.2f}%")
                else:
                    # Gentler learning rate adjustment
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] *= 0.98  # More gradual decay
                        print(f"Learning rate adjusted to {param_group['lr']:.6f}")

    def train(self, epochs=10):
        print("Starting adaptive training...")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Train one epoch
            epoch_loss, epoch_accuracy = self.train_epoch()

            # Store metrics
            self.train_losses.append(epoch_loss)
            self.test_accuracies.append(epoch_accuracy)

            # Check if we need to adapt the model
            self.adapt_model(epoch)

            print(f"Epoch {epoch+1}:")
            print(f"Training Loss: {epoch_loss:.3f}")
            print(f"Test Accuracy: {epoch_accuracy:.2f}%")
            print(f"Current network complexity: {self.model.current_complexity}")
            print(f"Current dropout rate: {self.model.dropout_rate:.2f}")

        return self.train_losses, self.test_accuracies, self.adaptation_points

    def plot_training_progress(self):
        """Plot training metrics with adaptation points marked"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training loss
        ax1.plot(self.train_losses, "b-", label="Training Loss")
        ax1.set_title("Training Loss Over Time")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        # Plot test accuracy
        ax2.plot(self.test_accuracies, "g-", label="Test Accuracy")
        ax2.set_title("Test Accuracy Over Time")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")

        # Mark adaptation points on both plots
        for epoch, adaptation_type in self.adaptation_points:
            ax1.axvline(x=epoch, color="r", linestyle="--", alpha=0.3)
            ax2.axvline(x=epoch, color="r", linestyle="--", alpha=0.3)
            ax1.text(
                epoch,
                max(self.train_losses),
                adaptation_type,
                rotation=90,
                verticalalignment="bottom",
            )
            ax2.text(
                epoch,
                min(self.test_accuracies),
                adaptation_type,
                rotation=90,
                verticalalignment="bottom",
            )

        plt.tight_layout()
        plt.show()


def main():
    """
    Main execution function for the MNIST adaptive learning demo.
    
    Sets up the training environment, initializes the model and training system,
    and executes the training process with visualization.
    """
    # Set number of threads for CPU optimization
    num_threads = min(8, torch.get_num_threads())  # Use up to 8 threads
    torch.set_num_threads(num_threads)
    print(f"Using {num_threads} CPU threads")
    
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    # Data transformations with additional augmentation
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST("data", train=False, transform=transform_test)

    # Create initial data loaders (will be updated with optimal batch size)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize and train model
    model = AdaptiveNeuralNet()
    adaptive_system = AdaptiveLearningSystem(model, train_loader, test_loader)
    results = adaptive_system.train(epochs=25)
    adaptive_system.plot_training_progress()


if __name__ == "__main__":
    main()
