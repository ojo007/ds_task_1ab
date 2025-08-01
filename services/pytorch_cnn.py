import torch
import torch.nn as nn

class PyTorchCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(PyTorchCNN, self).__init__()

        # Feature extraction layers (Convolutional)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier layers (Fully Connected)
        # This structure MUST EXACTLY match the training script
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128 * 16 * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps to a vector
        x = self.classifier(x)
        return x
