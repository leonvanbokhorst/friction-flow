import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import random
import pandas as pd

class NameDataset(Dataset):
    """Dataset for processing name characters into training samples."""
    
    def __init__(self, names: List[str], context_size: int = 3):
        # Clean names to only include valid string names
        self.names = []
        for name in names:
            # Skip non-string/NaN values
            if pd.isna(name):
                continue
            # Convert to string if needed
            name_str = str(name)
            cleaned = ''.join(c for c in name_str.lower() if c.isalpha())
            if cleaned:  # Only add non-empty names
                self.names.append(cleaned)
        
        self.context_size = context_size
        
        # Build character set from cleaned names first
        chars = set(''.join(self.names))
        
        # Then add special tokens
        self.special_tokens = ['<S>', '<E>']
        chars.update(self.special_tokens)
        self.chars = sorted(list(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.X, self.y = self._build_dataset()
    
    def _build_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create context/target pairs from names."""
        X, y = [], []
        for name in self.names:
            # Create context with start tokens
            context = ['<S>'] * self.context_size
            
            # Process each character in the name
            for char in name:
                X.append([self.stoi[c] for c in context])
                y.append(self.stoi[char])
                context = context[1:] + [char]
            
            # Add the end token as a single unit
            X.append([self.stoi[c] for c in context])
            y.append(self.stoi['<E>'])
        
        return torch.tensor(X), torch.tensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class NameGenerator(nn.Module):
    """Neural network for generating names."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 24, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                epochs: int = 50,
                learning_rate: float = 0.01) -> List[float]:
    """Train the name generation model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return losses

def generate_name(model: nn.Module, 
                 dataset: NameDataset, 
                 max_length: int = 20) -> str:
    """Generate a new name using the trained model."""
    model.eval()
    with torch.no_grad():
        context = torch.tensor([[dataset.stoi['<S>']] * dataset.context_size])
        name = []
        
        while True:
            output = model(context)
            probs = torch.softmax(output, dim=1)
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = dataset.itos[next_char_idx]
            
            if next_char == '<E>' or len(name) >= max_length:
                break
                
            name.append(next_char)
            context = torch.cat([context[:, 1:], 
                               torch.tensor([[next_char_idx]])], dim=1)
    
    return ''.join(name)

def generate_unique_names(model: nn.Module, 
                         dataset: NameDataset, 
                         num_names: int = 10,
                         max_attempts: int = 10,
                         max_length: int = 20) -> List[str]:
    """Generate unique names using the trained model."""
    generated_names = set()
    attempts = 0
    
    while len(generated_names) < num_names and attempts < max_attempts:
        name = generate_name(model, dataset, max_length)
        if name and name not in generated_names:
            generated_names.add(name)
        attempts += 1
    
    return list(generated_names)

if __name__ == "__main__":

    df = pd.read_csv("pocs/names/kids_names.csv")
    dataset = NameDataset(df['name'].tolist())
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = NameGenerator(len(dataset.chars))
    losses = train_model(model, train_loader)

    # Generate and analyze unique names
    print("\nGenerated Names:")
    generated_names = generate_unique_names(model, dataset, num_names=10)

    # Print statistics
    print("\nGeneration Statistics:")
    print(f"Unique names generated: {len(generated_names)}")
    print(f"Average name length: {sum(len(name) for name in generated_names) / len(generated_names):.1f}")
    print("\nGenerated names:")
    for name in generated_names:
        print(f"- {name}")

    # Check if generated names exist in training data
    training_names = set(dataset.names)
    existing_names = [name for name in generated_names if name in training_names]
    if existing_names:
        print("\nNames that exist in training data:")
        for name in existing_names:
            print(f"- {name}")
    else:
        print("\nAll generated names are unique from training data!")
