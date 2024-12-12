import torch
from torch import Tensor, nn
import logging

logger = logging.getLogger(__name__)  # Use the current module's name
logger.propagate = True

### Define transformer_classifier
class transformer_classifier(nn.Module):
    def __init__(self, input_size:int, n_channels:int, model_hyp:dict, classes:int):
        super(transformer_classifier, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_hyp['d_model'],
                                                        nhead=model_hyp['n_head'], batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=model_hyp['n_layer'])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(model_hyp['d_model']*n_channels, classes)

    def forward(self, x):
        z = self.transformer_encoder(x)
        logger.debug(f"transformer output size: {z.shape}")
        z = self.flatten(z)
        logger.debug(f"flatten output size: {z.shape}")
        y = self.linear(z)
        logger.debug(f"linear output size: {y.shape}")
        return y

### Define one layer MLP
class MLP_1l(nn.Module):
    def __init__(self, n_channels:int, d_model:int, classes:int):
        super(MLP_1l, self).__init__()
        self.mlp = nn.Sequential(

                nn.Linear(n_channels*d_model, classes),
            )
    
    def forward(self, x):
        
        z = torch.flatten(x, 1)
        z = self.mlp(z)
        return y

    
class MLP_3l(nn.Module):
    def __init__(self, n_channels:int, d_model:int, classes:int):
        super(MLP_3l, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(n_channels*d_model, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, classes),
            )
    
    def forward(self, x):
        
        z = torch.flatten(x, 1)
        z = self.mlp(z)
        return y