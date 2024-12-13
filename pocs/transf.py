"""
Transformer-based Addition Learning Model
=======================================

This module implements a transformer architecture specialized for learning arithmetic
operations, specifically addition. It serves as an educational example of applying
attention mechanisms to a concrete, deterministic task.

Historical Context:
-----------------
The transformer architecture, introduced by Vaswani et al. in "Attention Is All You Need" (2017),
revolutionized natural language processing. While originally designed for NLP tasks,
this implementation demonstrates its versatility by applying it to mathematical operations.
This follows a tradition of using neural networks for arithmetic, dating back to early
experiments by Hoshen & Peleg (2016) and others.

Key Components:
-------------
1. PositionalEncoding: Implements the sinusoidal position encoding from the original
   transformer paper, allowing the model to understand sequence order.
2. AdditionTransformer: A specialized transformer that learns to perform addition
   through self-attention mechanisms.
3. Data Generation: Custom utilities for creating training data of addition problems.

Technical Implementation:
----------------------
- Uses PyTorch's nn.TransformerEncoder as the core architecture
- Employs scaled dot-product attention mechanisms
- Implements position-wise feed-forward networks
- Uses dropout and layer normalization for regularization

References:
----------
1. Vaswani et al. (2017). "Attention Is All You Need"
   https://arxiv.org/abs/1706.03762
   
2. Hoshen & Peleg (2016). "Visual Learning of Arithmetic Operations"
   https://arxiv.org/abs/1506.02264

3. PyTorch Transformer Documentation
   https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

Usage Example:
------------
>>> model = AdditionTransformer()
>>> model = train_example()
>>> test_addition(model, 123, 456)
123 + 456 = 579 (expected 579)

Notes:
-----
This implementation is designed for educational purposes and demonstrates:
- Transformer architecture principles
- Sequence-to-sequence learning
- Position encoding
- Multi-head attention mechanisms
- PyTorch neural network development
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in 'Attention Is All You Need'.
    
    This encoding adds position-dependent patterns to the input embeddings,
    allowing the model to understand the sequential nature of the input.
    
    Args:
        d_model (int): The dimension of the model's embeddings
        max_len (int): Maximum sequence length to pre-compute encodings for
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        # Create position matrix [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        # Create division terms for sin/cos functions
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        # Fill even indices with sin and odd indices with cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Reshape for batch processing
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register as buffer (persistent state)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim]
        Returns:
            Tensor: Input with positional encoding added
        """
        x = x.transpose(0, 1)  # Transpose for sequence first format
        x = x + self.pe[: x.size(0)]  # Add positional encoding
        return x.transpose(0, 1)  # Transpose back to batch first format


class AdditionTransformer(nn.Module):
    """
    A Transformer model specialized for learning addition operations.
    
    This model learns to perform addition by processing digit sequences
    through self-attention mechanisms.
    
    Args:
        vocab_size (int): Size of vocabulary (digits 0-9 plus special tokens)
        d_model (int): Dimension of the model's embeddings
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer encoder layers
        max_len (int): Maximum sequence length
        dropout (float): Dropout rate for regularization
    """
    def __init__(self, vocab_size=12, d_model=128, nhead=8, num_layers=4, max_len=20, dropout=0.1):
        super().__init__()
        
        # Embedding layer converts digit tokens to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding adds position information
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # Configure transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Output decoder network
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, src):
        """
        Process input through the transformer model.
        
        Args:
            src (Tensor): Input tensor of digit tokens
            
        Returns:
            Tensor: Predicted digit probabilities
        """
        # Create mask for padding tokens
        padding_mask = (src == 0)
        
        # Convert tokens to embeddings and scale
        x = self.embedding(src) * np.sqrt(self.embedding.embedding_dim)
        x = self.dropout(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Process through transformer
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Decode to vocabulary size
        output = self.decoder(output)

        return output


def generate_addition_data(num_samples, max_digits=3):
    """
    Generates training data for addition problems.
    
    Creates pairs of numbers and their sums, formatted as sequences
    suitable for transformer input.
    
    Args:
        num_samples (int): Number of addition problems to generate
        max_digits (int): Maximum number of digits in each number
        
    Returns:
        tuple: (input_tensor, target_tensor) for training
    """
    def int_to_list(n, max_digits):
        """Convert integer to list of digits with padding"""
        return [int(d) for d in str(n).zfill(max_digits)]

    X, y = [], []
    for _ in range(num_samples):
        # Generate random numbers
        num1 = np.random.randint(0, 10**max_digits)
        num2 = np.random.randint(0, 10**max_digits)

        # Convert to digit lists
        num1_list = int_to_list(num1, max_digits)
        num2_list = int_to_list(num2, max_digits)

        # Calculate sum and format output
        sum_val = num1 + num2
        sum_list = int_to_list(sum_val, max_digits + 1)
        # Pad output to match input length
        sum_list = sum_list + [0] * (len(num1_list) + 1 + len(num2_list) - len(sum_list))

        # Format input with separator token
        x = num1_list + [10] + num2_list  # 10 is separator token

        X.append(x)
        y.append(sum_list)

    return torch.tensor(X), torch.tensor(y)


# Voorbeeld gebruik:
def train_example():
    # Model instantiëren
    model = AdditionTransformer(
        vocab_size=12,  # 0-9 plus start/end tokens
        d_model=256,    # Increased model capacity
        nhead=8,
        num_layers=4,
        max_len=20      # Reduced max_len since we don't need 1000
    )

    # Data genereren
    X_train, y_train = generate_addition_data(5000, max_digits=3)  # More training data
    
    # Create DataLoader with smaller batch size
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Loss en optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)  # Adjusted learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Training loop
    model.train()
    best_loss = float('inf')
    for epoch in range(30):  # More epochs
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            output = model(batch_X)
            output = output.reshape(-1, output.size(-1))
            target = batch_y.reshape(-1)
            
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/30], Average Loss: {avg_loss:.4f}")

    return model


# Test function for better evaluation
def test_addition(model, num1, num2, max_digits=3):
    model.eval()
    with torch.no_grad():
        num1_list = [int(d) for d in str(num1).zfill(max_digits)]
        num2_list = [int(d) for d in str(num2).zfill(max_digits)]
        test_input = torch.tensor([num1_list + [10] + num2_list])
        
        output = model(test_input)
        predicted = torch.argmax(output, dim=-1)
        
        # Filter out padding tokens
        result = ''.join(str(x) for x in predicted[0].tolist() if x != 0).lstrip('0')
        if not result:  # If result is empty after filtering
            result = '0'
        
        print(f"{num1} + {num2} = {result} (expected {num1 + num2})")
        return result


if __name__ == "__main__":
    model = train_example()

    # Test multiple examples
    test_addition(model, 123, 456)
    test_addition(model, 45, 67)
    test_addition(model, 789, 123)
