"""
CNN-GRU model for audio classification
"""
import torch
import torch.nn as nn


class SpectrogramCNN_GRUNet(nn.Module):
    """CNN-GRU model for spectrogram classification"""
    
    def __init__(self, num_classes=2):
        super(SpectrogramCNN_GRUNet, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop = nn.Dropout(0.25)
        
        # GRU Layers
        self.gru_input_size = 128 * 8
        self.gru1 = nn.GRU(
            input_size=self.gru_input_size, 
            hidden_size=68, 
            batch_first=True, 
            num_layers=1
        )
        self.gru2 = nn.GRU(
            input_size=68, 
            hidden_size=68, 
            batch_first=True, 
            num_layers=1
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(68, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.drop_fc = nn.Dropout(0.5)

    def forward(self, x):
        # CNN Feature Extraction
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop(x)

        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.drop(x)

        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop(x)

        # Reshape for GRU
        b, c, h, w = x.size()
        x = x.view(b, w, c * h)

        # GRU Sequence Modeling
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        # Use final hidden state
        x = x[:, -1, :]

        # Fully Connected Layer
        x = torch.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return x