import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Union, Tuple, Callable

class FlexibleLSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the Flexible LSTM Network.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features/classes.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate to apply between LSTM layers (default: 0.0).
        """
        super(FlexibleLSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        lstm_out, _ = self.lstm(x)  # Output and hidden states
        last_out = lstm_out[:, -1, :]  # Get the output of the last time step
        output = self.fc(last_out)    # Fully connected layer
        return output

# Example Usage
def main():
    # Define the parameters
    input_size = 10   # Number of input features
    output_size = 3   # Number of output classes
    hidden_size = 64  # Hidden state size
    num_layers = 2    # Number of LSTM layers
    dropout = 0.2     # Dropout rate

    # Create the model
    model = FlexibleLSTMNet(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
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
