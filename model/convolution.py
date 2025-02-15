import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (1,28,28) -> (32,28,28)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (32,28,28) -> (64,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # (64,28,28) -> (64,14,14)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (64,14,14) -> (128,14,14)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# (128,14,14) -> (128,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        # (128,14,14) -> (128,7,7)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),  # Fully Connected Layer
            nn.ReLU(),
            nn.Linear(512, 10)            # Output Layer
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
    