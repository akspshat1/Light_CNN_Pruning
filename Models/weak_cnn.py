import torch
import torch.nn as nn

class WeakCNN(nn.Module):
    def __init__(self):
        super(WeakCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        # Block 1
        x = self.pool1(self.relu1(self.conv1(x)))
        # print(x.shape)
        # Block 2
        x = self.pool2(self.relu2(self.conv2(x)))
        # print(x.shape)

        # Flatten before fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        return x
