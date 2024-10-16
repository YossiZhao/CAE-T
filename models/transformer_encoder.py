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

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_hyp['d_model'],
                                                        nhead=model_hyp['n_head'], batch_first=True)
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
        logger.debug(f"transformer output size: {z.shape}")
        z = self.flatten(z)
        # z = self.flatten(z)
        logger.debug(f"flatten output size: {z.shape}")
        y = self.linear(z)
        logger.debug(f"linear output size: {y.shape}")
        return y