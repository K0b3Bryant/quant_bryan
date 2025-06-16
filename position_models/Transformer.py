import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Union, Tuple, Callable

class FlexibleTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the Flexible Transformer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features/classes.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            hidden_dim (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate (default: 0.1).
        """
        super(FlexibleTransformer, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer."""
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Pool across the sequence dimension
        output = self.fc(x)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Positional encoding module.

        Args:
            d_model (int): The embedding dimension.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input tensor."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Example Usage
def main():
    # Define the parameters
    input_size = 10    # Number of input features
    output_size = 3    # Number of output classes
    num_heads = 2      # Number of attention heads
    num_layers = 2     # Number of transformer layers
    hidden_dim = 64    # Hidden dimension size
    dropout = 0.1      # Dropout rate

    # Create the model
    model = FlexibleTransformer(
        input_size=input_size,
        output_size=output_size,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    # Print model architecture
    print(model)

    # Example training setup
    batch_size = 5
    sequence_length = 15
    x = torch.rand((batch_size, sequence_length, input_size))  # Random input tensor
    y = torch.randint(0, output_size, (batch_size,))           # Random target tensor

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    print(f"Initial Loss: {loss.item()}")

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    main()
