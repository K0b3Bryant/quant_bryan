import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Union, Tuple, Callable

class FlexibleNeuralNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int],
        activation: Union[str, Callable] = 'ReLU',
        dropout: float = 0.0,
    ):
        """
        Initialize the Flexible Neural Network.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features/classes.
            hidden_layers (List[int]): List defining the sizes of hidden layers.
            activation (Union[str, Callable]): Activation function to use (e.g., 'ReLU', 'Sigmoid', etc.) or a callable.
            dropout (float): Dropout rate to apply after each layer (default: 0.0).
        """
        super(FlexibleNeuralNet, self).__init__()

        # Map string to activation function if necessary
        self.activation = self.get_activation(activation)
        
        layers = []
        previous_size = input_size

        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(self.activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size

        # Output layer
        layers.append(nn.Linear(previous_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if x.shape[1] != self.network[0].in_features:
            raise ValueError(f"Expected input with {self.network[0].in_features} features, but got {x.shape[1]}.")
        return self.network(x)

    @staticmethod
    def get_activation(activation: Union[str, Callable]) -> Callable:
        """Get the activation function."""
        if isinstance(activation, str):
            activations = {
                'ReLU': nn.ReLU(),
                'Sigmoid': nn.Sigmoid(),
                'Tanh': nn.Tanh(),
                'LeakyReLU': nn.LeakyReLU(),
                'Softmax': nn.Softmax(dim=1),
                'ELU': nn.ELU()
            }
            return activations.get(activation, nn.ReLU())
        elif callable(activation):
            return activation
        else:
            raise ValueError("Invalid activation function. Use a string or callable.")

# Example Usage
def main():
    # Define the parameters
    input_size = 20
    output_size = 3
    hidden_layers = [64, 128, 64]
    activation = 'ReLU'
    dropout = 0.2

    # Create the model
    model = FlexibleNeuralNet(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout
    )

    # Print model architecture
    print(model)

    # Example training setup
    x = torch.rand((5, input_size))
    y = torch.randint(0, output_size, (5,))

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
