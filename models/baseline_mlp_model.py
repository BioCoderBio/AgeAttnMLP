import torch
import torch.nn as nn

class BasicMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model.
    This model serves as a baseline for comparison.
    """
    def __init__(self, input_dim=1280, dropout=0.25):
        """
        Initializes the BasicMLP model.

        Args:
            input_dim (int): The dimensionality of the input features.
            dropout (float): The dropout rate to use.
        """
        super().__init__()
        
        self.net = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Block 2
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Block 3
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Block 4
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Head
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.net(x)
        return out.squeeze(-1)
