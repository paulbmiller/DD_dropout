"""
CNN with 3 conv layers and a fully connected classification layer
"""

import torch.nn as nn


class BasicLinear(nn.Module):
    """
    Simple feed forward linear neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    lin1 : torch.nn.Sequential
    lin2 : torch.nn.Sequential
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        """
        Creates a BasicLinear model from scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(BasicLinear, self).__init__()

        self.expected_input_size = (32, 32)

        # First layer
        self.lin1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        # Second layer
        self.lin2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(1024, output_channels)
        )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.fc(x)
        return x
