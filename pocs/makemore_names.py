import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import pandas as pd

class NameDataset(Dataset):
    """A PyTorch Dataset for processing character-level name data.

    This dataset preprocesses names for character-level language modeling, creating
    context windows and corresponding target characters for training. It handles
    data cleaning, vocabulary creation, and conversion between characters and indices.

    Args:
        names (List[str]): List of input names to process
        context_size (int, optional): Number of previous characters to use as context. Defaults to 3.

    Attributes:
        names (List[str]): Cleaned and normalized names
        context_size (int): Size of the context window
        chars (List[str]): List of all unique characters in vocabulary
        special_tokens (List[str]): Special tokens for sequence start/end
        stoi (Dict[str, int]): String to index mapping
        itos (Dict[int, str]): Index to string mapping
        X (torch.Tensor): Processed context windows
        y (torch.Tensor): Corresponding target characters
    """
    
    def __init__(self, names: List[str], context_size: int = 3):
        # Initialize dataset with list of names and context window size
        # Context size determines how many previous characters to consider when predicting the next one
        
        # Clean and normalize the input names
        self.names = []
        for name in names:
            # Handle NaN values that might come from pandas DataFrame
            if pd.isna(name):
                continue
            # Convert to lowercase and remove non-alphabetic characters
            name_str = str(name)
            cleaned = ''.join(c for c in name_str.lower() if c.isalpha())
            if cleaned:  # Only add non-empty names
                self.names.append(cleaned)
        
        self.context_size = context_size
        
        # Create vocabulary:
        # 1. First collect all unique characters from names
        # 2. Add special tokens for sequence start (<S>) and end (<E>)
        # 3. Create bidirectional mappings between characters and indices
        chars = set(''.join(self.names))
        self.special_tokens = ['<S>', '<E>']
        chars.update(self.special_tokens)
        self.chars = sorted(list(chars))
        # stoi: string-to-index mapping
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        # itos: index-to-string mapping
        self.itos = {i: c for i, c in enumerate(self.chars)}
        
    def _build_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates training pairs of context windows and target characters.

        Processes each name to create sliding context windows and their corresponding
        target characters. Handles special tokens for sequence start and end.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - X: Tensor of context windows (shape: [n_samples, context_size])
                - y: Tensor of target character indices (shape: [n_samples])
        """
        X, y = [], []  # X: contexts, y: target characters
        for name in self.names:
            # Start each name with context_size number of start tokens
            # Example: for "john" with context_size=3: ["<S>", "<S>", "<S>"] -> predict "j"
            context = ['<S>'] * self.context_size
            
            # For each character in name:
            # 1. Use current context to predict next character
            # 2. Slide context window by removing oldest char and adding current char
            for char in name:
                X.append([self.stoi[c] for c in context])  # Convert context chars to indices
                y.append(self.stoi[char])  # Convert target char to index
                context = context[1:] + [char]  # Slide context window
            
            # After processing name, predict end token
            X.append([self.stoi[c] for c in context])
            y.append(self.stoi['<E>'])
        
        return torch.tensor(X), torch.tensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class NameGenerator(nn.Module):
    """Neural network model for generating names using LSTM architecture.

    Implements a character-level language model using embeddings and LSTM layers
    to learn patterns in names and generate new ones.

    Args:
        vocab_size (int): Size of the character vocabulary
        embedding_dim (int, optional): Dimension of character embeddings. Defaults to 24.
        hidden_dim (int, optional): Dimension of LSTM hidden state. Defaults to 128.

    Attributes:
        embedding (nn.Embedding): Embedding layer for characters
        lstm (nn.LSTM): LSTM layer for sequence processing
        fc (nn.Linear): Final linear layer for character prediction
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 24, hidden_dim: int = 128):
        super().__init__()
        # Character embedding layer: converts character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layer processes sequence of embeddings
        # batch_first=True means input shape is (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # Final linear layer converts LSTM output to vocabulary probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of character indices
                Shape: [batch_size, sequence_length]

        Returns:
            torch.Tensor: Output logits for next character prediction
                Shape: [batch_size, vocab_size]
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                epochs: int = 50,
                learning_rate: float = 0.01) -> List[float]:
    """Trains the name generation model.

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader containing training data
        epochs (int, optional): Number of training epochs. Defaults to 50.
        learning_rate (float, optional): Learning rate for optimization. Defaults to 0.01.

    Returns:
        List[float]: List of average losses per epoch

    Note:
        Uses CrossEntropyLoss and Adam optimizer for training.
        Prints progress every 10 epochs.
    """
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

def generate_name(model: nn.Module, dataset: NameDataset, max_length: int = 20) -> str:
    """Generates a new name using the trained model.

    Uses autoregressive generation to create new names character by character,
    sampling from the model's predicted probability distribution.

    Args:
        model (nn.Module): Trained name generation model
        dataset (NameDataset): Dataset containing vocabulary mappings
        max_length (int, optional): Maximum length of generated name. Defaults to 20.

    Returns:
        str: Generated name

    Note:
        Generation stops when either:
        1. The end token '<E>' is generated
        2. The maximum length is reached
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        # Initialize context with start tokens
        context = torch.tensor([[dataset.stoi['<S>']] * dataset.context_size])
        name = []
        
        while True:
            # Get model predictions for next character
            output = model(context)
            # Convert logits to probabilities
            temperature = 0.8  # Lower = more conservative, higher = more creative
            scaled_logits = output / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            # Sample next character index from probability distribution
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = dataset.itos[next_char_idx]
            
            # Stop if end token or max length reached
            if next_char == '<E>' or len(name) >= max_length:
                break
                
            name.append(next_char)
            # Update context window for next prediction
            context = torch.cat([context[:, 1:], 
                               torch.tensor([[next_char_idx]])], dim=1)
    
    return ''.join(name)

def generate_unique_names(model: nn.Module, 
                         dataset: NameDataset, 
                         num_names: int = 10,
                         max_attempts: int = 10,
                         max_length: int = 20) -> List[str]:
    """Generates multiple unique names using the trained model.

    Args:
        model (nn.Module): Trained name generation model
        dataset (NameDataset): Dataset containing vocabulary mappings
        num_names (int, optional): Number of unique names to generate. Defaults to 10.
        max_attempts (int, optional): Maximum generation attempts. Defaults to 10.
        max_length (int, optional): Maximum length of each name. Defaults to 20.

    Returns:
        List[str]: List of generated unique names

    Note:
        May return fewer names than requested if max_attempts is reached
        before generating enough unique names.
    """
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
