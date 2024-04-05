from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cldnn.py
class CLDNN(ModelBase):
    """

    References
        X. Liu, D. Yang, and A. E. Gamal, “Deep neural network architectures for modulation classification,” 
        in 2017 51st Asilomar Conference on Signals, Systems, and Computers, Oct. 2017, pp. 915-919. 
        doi: 10.1109/ACSSC.2017.8335483.

    """

    def __init__(
        self,
        input_samples: int,
        input_channels: int,
        classes: List[str],
        learning_rate: float = 0.001,
        **kwargs
    ):
        super().__init__(classes=classes, **kwargs)

        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate
        self.example_input_array = torch.zeros((1,input_channels,input_samples), dtype=torch.cfloat)

        # Batch x 1-channel x input_samples x IQ 
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=256,
                kernel_size=(3,1),
                padding='same',
            ),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3,2),
                padding='same',
            ),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(
                in_channels=256,
                out_channels=80,
                kernel_size=(3,1),
                padding='same',
            ),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(
                in_channels=80,
                out_channels=80,
                kernel_size=(3,1),
                padding='same',
            ),
            nn.Dropout2d(0.5),
        )


        self.gru = nn.LSTM(
            input_size=80*2,  # 80 channels * IQ (2)
            hidden_size=50,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
            dropout=0.6
        )

        # Batch x Features
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout1d(0.5),
            nn.Linear(128, len(classes)),
        )


    def forward(self, x: torch.Tensor, **kwargs):
        x = torch.view_as_real(x)

        # Convolutional feature extraction layers
        x = self.conv(x)

        # Flatten along channels and I/Q, preserving time dimension
        x = x.transpose(1, 2) # BxCxTxIQ -> # BxTxCxIQ
        x = torch.flatten(x, 2).contiguous() # BxTxCxIQ -> BxTxF

        # Temporal feature extraction
        x, _ = self.gru(x)

        # MLP Classification stage
        y = self.mlp(x)

        return y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)