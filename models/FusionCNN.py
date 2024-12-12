import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)  # Use the current module's name

__all__ = [
    "EEG_CNN",
    "FusionNetwork",
    "EEG_Pathology_Detection"
]


class EEG_CNN(nn.Module):
    def __init__(self, input_channels=19, classes=2, stage=1):
        super(EEG_CNN, self).__init__()
        self.stage = stage
        # First Layer: Temporal (2D Conv with kernel 1x10) + Spatial (2D Conv with kernel 19x1)
        self.conv1_1 = nn.Conv2d(1, 50, kernel_size=(1, 10), stride=(1, 1), padding=(0, 0))  # Temporal
        self.conv1_2 = nn.Conv2d(50, 50, kernel_size=(input_channels, 1), stride=(1, 1), padding=(0, 0))  # Spatial
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))  # Pooling along time axis

        # Second Layer: 1D convolution
        self.conv2 = nn.Conv1d(50, 100, kernel_size=10, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        # Third Layer: 1D convolution
        self.conv3 = nn.Conv1d(100, 100, kernel_size=10, stride=1, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)

        # Fourth Layer: 1D convolution
        self.conv4 = nn.Conv1d(100, 200, kernel_size=10, stride=1, padding=0)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=3)
        
        # Fully Connected Layers for Stage 1
        if self.stage == 1:
            self.fc1 = nn.Linear(200 * 143, 2048)  # Adjust size based on pool4 output
            self.fc2 = nn.Linear(2048, classes)
            self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input for 2D convolutions (batch, channels=1, height=input_channels, width=time_length)
        x = x.unsqueeze(1)  # Shape: (batch, 1, 19, 12000)
        logger.debug(f'x.unsqueeze: {x.shape}')

        # First Layer: Temporal convolution (along time axis)
        temporal = self.relu(self.conv1_1(x))  # Shape: (batch, 50, 19, time_length - 9)
        logger.debug(f'x.conv1_1(temporal): {temporal.shape}')

        # First Layer: Spatial convolution (along channel axis)
        spatial = self.relu(self.conv1_2(temporal))  # Shape: (batch, 50, 1, time_length - 9)
        logger.debug(f'x.conv1_2(spatial): {spatial.shape}')
        
        spatial = spatial.squeeze(2)  # Remove the spatial height dimension (batch, 50, time_length - 9)
        logger.debug(f'x.after squeeze: {spatial.shape}')

        # First pooling
        pool1 = self.pool1(spatial.unsqueeze(2)).squeeze(2)  # Shape: (batch, 50, reduced_time_length)

        # Second Layer: 1D convolution
        pool2 = self.pool2(self.relu(self.conv2(pool1)))  # Shape: (batch, 100, reduced_time_length)

        # Third Layer: 1D convolution
        pool3 = self.pool3(self.relu(self.conv3(pool2)))  # Shape: (batch, 100, reduced_time_length)

        # Fourth Layer: 1D convolution
        pool4 = self.pool4(self.relu(self.conv4(pool3)))  # Shape: (batch, 200, reduced_time_length)
        
        if self.stage == 1:
            # Flatten output from pool4 for Stage 1
            pool4_flat = pool4.view(pool4.size(0), -1)  # Shape: (batch, 200 * 144)

            # Fully connected layers
            fc1_out = self.relu(self.fc1(pool4_flat))
            fc2_out = self.relu(self.fc2(fc1_out))
            output = self.softmax(fc2_out)
            return output

        return pool1, pool2, pool3, pool4
    
class FusionNetwork(nn.Module):
    def __init__(self, input_sizes, fusion_type="MLP", hidden_layers=[8192, 4096], classes=2):
        super(FusionNetwork, self).__init__()

        self.fusion_type = fusion_type

        # Concatenate and flatten input from all pooling layers
        total_input_size = sum(input_sizes)

        if fusion_type == "MLP":
            layers = []
            layers.append(nn.Linear(total_input_size, hidden_layers[0]))
            layers.append(nn.ReLU())
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_layers[-1], classes))  # Output layer
            self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)  # Concatenate flattened outputs
        if self.fusion_type == "MLP":
            y = self.mlp(x)
            logger.debug(f'after MLP: {y.shape}')
            return y
        
class EEG_Pathology_Detection(nn.Module):
    def __init__(self, cnn_model, fusion_model):
        super(EEG_Pathology_Detection, self).__init__()
        self.cnn = cnn_model
        self.fusion = fusion_model

    def forward(self, x):
        # Get outputs from CNN's pooling layers
        pool1, pool2, pool3, pool4 = self.cnn(x)

        # Flatten outputs from pooling layers
        pool1_flat = pool1.view(pool1.size(0), -1)
        pool2_flat = pool2.view(pool2.size(0), -1)
        pool3_flat = pool3.view(pool3.size(0), -1)
        pool4_flat = pool4.view(pool4.size(0), -1)

        # Pass flattened outputs to the fusion network
        return self.fusion([pool1_flat, pool2_flat, pool3_flat, pool4_flat])
    
    
    
    
# # Initialize the models
# cnn_model = EEG_CNN(input_channels=19, classes=2)
# fusion_model = FusionNetwork(input_sizes=[50 * 3997, 100 * 1329, 100 * 440, 200 * 143], fusion_type="MLP", hidden_layers=[8192, 4096], classes=2)
# model = EEG_Pathology_Detection(cnn_model, fusion_model)

# # debug

# # Forward pass
# pool1, pool2, pool3, pool4 = cnn_model(x)



# # Forward pass
# output = model(x)
# print(output.shape)  # Expected output shape: (32, 2)

# # FLOPs
# flops = FlopCountAnalysis(model, x)
# print("FLOPs:", flops.total())
# print(parameter_count_table(model))