"""
Simple network with 3 conv layer with max pooling and 1 fully connected classification layer
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

class CNN_CIFAR(nn.Module):
    """
    Simple feed forward neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
    fc : torch.nn.Sequential
        Final classification fully connected layer

    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        """
        Creates a CNN_CIFAR model from scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(CNN_CIFAR, self).__init__()

        self.expected_input_size = (32, 32)

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=10, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d()
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(1024 * 2 * 2, output_channels)
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x
