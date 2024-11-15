import torch
import torch.nn as nn
import torch.optim as optim

#GENERATOR
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, dilation=2),  # Dilated conv
            nn.InstanceNorm1d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, dilation=2),
            nn.InstanceNorm1d(in_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)  # Add skip connection (residual learning)

"""Residual Neural Network"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class Generator(nn.Module):
    def __init__(self, input_size, output_size, seq_len):
        super(Generator, self).__init__()

        # First layer: Conv1d with kernel size 3, stride 1 (k3n64s1)
        self.initial_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Residual blocks (B residual blocks)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]  # 16 residual blocks
        )

        # Deconvolution layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Final output layer: Conv1d (k3n64s1)
        self.output_layer = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        x = self.deconv_layers(x)
        x = self.output_layer(x)
        return x
    
#DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, input_size, seq_len):
        super(Discriminator, self).__init__()

        # Define convolutional layers, following the architecture in the figure
        self.conv_layers = nn.Sequential(
            # First Conv Layer: n64s1 (input: 1 channel, output: 64 channels)
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Second Conv Layer: n64s2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Third Conv Layer: n128s1
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth Conv Layer: n128s2
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Fifth Conv Layer: n256s1
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Sixth Conv Layer: n256s2
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # Seventh Conv Layer: n512s1
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # Eighth Conv Layer: n512s2
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x
