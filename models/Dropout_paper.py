"""
CNN with 3 conv layers and a fully connected classification layer
"""

import torch.nn as nn


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Dropout_paper(nn.Module):
    """
    Simple feed forward convolutional neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(Dropout_paper, self).__init__()

        self.expected_input_size = (28, 28)

        self.lin1 = nn.Sequential(
            Flatten(),
            nn.Linear(28 * 28 * 3, 8192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.lin2 = nn.Sequential(
            nn.Linear(8192, 8192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(8192, output_channels)
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
