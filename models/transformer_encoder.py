import torch
from torch import Tensor, nn
import logging

logger = logging.getLogger(__name__)  # Use the current module's name
logger.propagate = True
# logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler()
# # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# # handler.setFormatter(formatter)
# logger.addHandler(handler)

### Define transformer_classifier
class transformer_classifier(nn.Module):
    def __init__(self, input_size:int, n_channels:int, model_hyp:dict, classes:int):
        super(transformer_classifier, self).__init__()
        
#         self.ae_1 = nn.Sequential(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, 
#                                      stride=3, kernel_size=7, dilation=1, groups=n_channels,
#                                             padding_mode='reflect'),
#                                  nn.BatchNorm1d(n_channels),
#                                  nn.ReLU())
        
#         self.ae_2 = nn.Sequential(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, 
#                                      stride=2, kernel_size=4, dilation=1, groups=n_channels,
#                                             padding_mode='reflect'),
#                                  nn.BatchNorm1d(n_channels),
#                                  nn.ReLU())

#         self.ae_3 = nn.Sequential(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, 
#                                      stride=2, kernel_size=4, dilation=1, groups=n_channels,
#                                             padding_mode='reflect'),
#                                  nn.BatchNorm1d(n_channels),
#                                  nn.ReLU())
#         self.hidden_size = 597    # need to be calculated every time if you change shape of input
#         self.ae = AutoEncoder(input_size=input_size, hidden_size=model_hyp['d_model'])  
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_hyp['d_model'],
                                                        nhead=model_hyp['n_head'],batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=model_hyp['n_layer'])
        self.norm = nn.LayerNorm([19,512])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(model_hyp['d_model']*n_channels, classes)

    def forward(self, x):
        # z = self.ae_1(x)
        # z = self.ae_2(z)
        # z = self.ae_3(z)
#         z = z[:, :, :1496] 
#         logger.debug(f"ae output size: %{z.shape}")
        z = self.transformer_encoder(x)
        logger.debug(f"transformer output size: %{z.shape}")
        z = self.flatten(z)
        # z = self.flatten(z)
        logger.debug(f"flatten output size: %{z.shape}")
        y = self.linear(z)
        return y