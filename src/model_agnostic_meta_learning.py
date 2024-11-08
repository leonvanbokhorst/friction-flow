from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class MetaModelGenerator(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        inner_lr: float = 0.05,
        meta_lr: float = 0.003,
    ):
        super().__init__()
        self.inner_lr = inner_lr
        self.input_size = input_size
        self.output_size = output_size

        # Simple architecture with skip connections
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
                for i in range(len(hidden_sizes) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        # Initialize weights with smaller values
        self.apply(self._init_weights)

        # Use SGD with momentum for meta-optimization
        self.meta_optimizer = optim.SGD(
            self.parameters(), lr=meta_lr, momentum=0.9, nesterov=True
        )

        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
            min_lr=1e-5,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        hidden = x

        for layer in self.hidden_layers:
            hidden = F.relu(layer(hidden) + hidden)  # Skip connection

        return self.output_layer(hidden)

    def meta_train_step(
        self,
        task_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Core MAML training step that implements the bi-level optimization:
        1. Inner Loop: Adapt to individual tasks using gradient descent
        2. Outer Loop: Update meta-parameters to optimize post-adaptation performance
        
        Args:
            task_batch: List of (support_x, support_y, query_x, query_y) tuples
                - support_x/y: Used for task adaptation (inner loop)
                - query_x/y: Used for meta-update (outer loop)
            device: Computation device (CPU/GPU)
        
        Returns:
            avg_meta_loss: Average loss across all tasks after adaptation
            avg_grad_norm: Average gradient norm for monitoring training
        """
        total_meta_loss = 0.0
        total_grad_norm = 0.0

        self.meta_optimizer.zero_grad()

        for support_x, support_y, query_x, query_y in zip(*task_batch):
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)

            fast_weights = {name: param.clone() for name, param in self.named_parameters()}
            # Multiple gradient steps for task adaptation
            for _ in range(3):  # Inner loop steps
                support_pred = self.forward_with_fast_weights(support_x, fast_weights)
                inner_loss = F.mse_loss(support_pred, support_y)

                # Compute gradients w.r.t fast_weights (create_graph=True enables higher-order gradients)
                grads = torch.autograd.grad(
                    inner_loss,
                    fast_weights.values(),
                    create_graph=True,  # Required for meta-learning
                    allow_unused=True,
                )

                # Update fast weights with gradient clipping
                for (name, weight), grad in zip(fast_weights.items(), grads):
                    if grad is not None:
                        clipped_grad = torch.clamp(grad, -1.0, 1.0)  # Stability
                        fast_weights[name] = weight - self.inner_lr * clipped_grad

            # Outer Loop: Meta-Update
            # Evaluate performance on query set using adapted weights
            query_pred = self.forward_with_fast_weights(query_x, fast_weights)
            meta_loss = F.mse_loss(query_pred, query_y)

            # Accumulate meta-gradients
            meta_loss.backward()  # This propagates through the entire inner loop
            total_meta_loss += meta_loss.item()

            # Monitor gradient norms
            with torch.no_grad():
                grad_norm = sum(
                    param.grad.norm().item() ** 2 
                    for param in self.parameters() 
                    if param.grad is not None
                ) ** 0.5
                total_grad_norm += grad_norm

        # Average and apply meta-update
        avg_meta_loss = total_meta_loss / len(task_batch)
        avg_grad_norm = total_grad_norm / len(task_batch)

        # Gradient clipping for stable meta-updates
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.meta_optimizer.step()

        return avg_meta_loss, avg_grad_norm

    def forward_with_fast_weights(
        self, x: torch.Tensor, fast_weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = F.relu(
            F.linear(
                x, fast_weights["input_layer.weight"], fast_weights["input_layer.bias"]
            )
        )
        hidden = x

        for i in range(len(self.hidden_layers)):
            next_hidden = F.relu(
                F.linear(
                    hidden,
                    fast_weights[f"hidden_layers.{i}.weight"],
                    fast_weights[f"hidden_layers.{i}.bias"],
                )
            )
            hidden = next_hidden + hidden  # Skip connection

        return F.linear(
            hidden,
            fast_weights["output_layer.weight"],
            fast_weights["output_layer.bias"],
        )


def create_synthetic_tasks(
    num_tasks: int = 100,
    samples_per_task: int = 50,
    input_size: int = 10,
    output_size: int = 1,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Creates a set of synthetic regression tasks for meta-learning training.
    Each task represents a different non-linear function with controlled complexity.
    
    Task Generation Process:
    1. Generate normalized input features
    2. Create task-specific transformation
    3. Add controlled noise for robustness
    4. Split into support (training) and query (testing) sets
    
    Returns:
        List of (support_x, support_y, query_x, query_y) tuples for each task
    """
    tasks = []
    split_idx = samples_per_task // 2  # 50/50 split between support and query sets

    for _ in range(num_tasks):
        # 1. Generate and normalize input features
        x = torch.randn(samples_per_task, input_size)
        x = (x - x.mean(0)) / (x.std(0) + 1e-8)  # Standardize inputs

        # 2. Create task-specific transformation
        coefficients = torch.randn(input_size, output_size) * 0.3  # Random linear transformation
        bias = torch.randn(output_size) * 0.1  # Random bias term

        # 3. Generate outputs with multiple non-linearities
        y = torch.matmul(x, coefficients) + bias  # Linear component
        y += 0.15 * torch.sin(2.0 * torch.matmul(x, coefficients))  # Sinusoidal component
        y += 0.08 * torch.tanh(1.5 * torch.matmul(x, coefficients))  # Tanh component

        # 4. Add adaptive noise based on signal magnitude
        noise_scale = 0.02 * torch.std(y)  # Noise proportional to output variance
        y += noise_scale * torch.randn_like(y)

        # 5. Split into support and query sets
        tasks.append((
            x[:split_idx],    # support_x: First half of inputs
            y[:split_idx],    # support_y: First half of outputs
            x[split_idx:],    # query_x:   Second half of inputs
            y[split_idx:]     # query_y:   Second half of outputs
        ))

    return tasks


def create_task_dataloader(
    tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_size: int = 4,
) -> DataLoader:
    """
    Organizes tasks into batches for efficient training.
    Shuffles tasks to prevent learning order dependencies.
    """
    # Reorganize tasks into separate lists
    support_x = [t[0] for t in tasks]
    support_y = [t[1] for t in tasks]
    query_x = [t[2] for t in tasks]
    query_y = [t[3] for t in tasks]

    # Create dataset and return DataLoader with shuffling
    dataset = list(zip(support_x, support_y, query_x, query_y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # Configuration
    INPUT_SIZE = 10
    OUTPUT_SIZE = 1
    HIDDEN_SIZES = [64, 64]  # Increased capacity
    BATCH_SIZE = 8  # Increased batch size
    NUM_TASKS = 200  # More tasks
    SAMPLES_PER_TASK = 100  # More samples per task

    # Create synthetic tasks with more controlled complexity
    tasks = create_synthetic_tasks(
        num_tasks=NUM_TASKS,
        samples_per_task=SAMPLES_PER_TASK,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
    )

    # Create task dataloader with fixed batch size
    task_dataloader = create_task_dataloader(tasks, batch_size=BATCH_SIZE)

    # Initialize meta-model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_model = MetaModelGenerator(
        input_size=INPUT_SIZE, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE
    ).to(device)

    # Training loop
    num_epochs = 20
    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 5  # Early stopping patience

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_grad_norm = 0.0

        for batch in task_dataloader:
            loss, grad_norm = meta_model.meta_train_step(batch, device)
            total_loss += loss
            total_grad_norm += grad_norm

        avg_loss = total_loss / len(task_dataloader)
        avg_grad_norm = total_grad_norm / len(task_dataloader)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}"
        )

        # Learning rate scheduling
        meta_model.scheduler.step(avg_loss)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Optional: Stop if loss is very low
        if avg_loss < 0.001:
            print("Reached target loss. Stopping training.")
            break

    # Generate new task-specific model
    new_task = create_synthetic_tasks(num_tasks=1)[0]
    support_x, support_y, query_x, query_y = new_task
    support_x, support_y = support_x.to(device), support_y.to(device)

    fast_weights = {
        name: param.clone() for name, param in meta_model.named_parameters()
    }
    # Quick adaptation
    for _ in range(5):
        support_pred = meta_model.forward_with_fast_weights(support_x, fast_weights)
        adapt_loss = F.mse_loss(support_pred, support_y)
        grads = torch.autograd.grad(
            adapt_loss, fast_weights.values(), create_graph=False
        )

        for (name, weight), grad in zip(fast_weights.items(), grads):
            if grad is not None:
                fast_weights[name] = weight - meta_model.inner_lr * grad

    # Evaluate on query set
    query_x, query_y = query_x.to(device), query_y.to(device)
    query_pred = meta_model.forward_with_fast_weights(query_x, fast_weights)
    final_loss = F.mse_loss(query_pred, query_y)
    print(f"\nNew Task Adaptation - Query Loss: {final_loss.item():.4f}")
