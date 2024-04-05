from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/Richardzhangxx/AMR-Benchmark/blob/main/RML201610a/DenseNet/rmlmodels/DenseNet.py
class DenseNet(ModelBase):
    """Convolutional Neural Network with skips

    Skips "are proposed to allow features learned from multiple layers to be effectively transmitted to the detection module."

    References
        Liu, X., Yang, D., El Gamal, A., 2017. Deep neural network architectures for modulation classification, 
        in: Proc. 51st Asilomar Conf. Signals, Syst., Comput., pp. 915-919.
        https://arxiv.org/pdf/1712.00443.pdf
        https://arxiv.org/pdf/1608.06993.pdf
    """

    def __init__(
        self,
        classes: List[str],
        input_channels: int=1,
        input_samples: int=1024,
        learning_rate: float = 0.001,
        **kwargs
    ):
        super().__init__(classes=classes, **kwargs)

        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate
        self.example_input_array = torch.zeros((1,input_channels,input_samples), dtype=torch.cfloat)

        # CNN Skip layers layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=(3,1), padding='same'),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,2), padding='same'),
        )

        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=80, kernel_size=(3,1), padding='same'),
        )

        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=592, out_channels=80, kernel_size=(3,1), padding='same'),
            nn.ReLU(),
        )

        # MLP layers
        dr = 0.6 # Dropout probability
        self.mlp = nn.Sequential(
            nn.Dropout2d(dr),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout1d(dr),
            nn.Linear(128, len(classes)),
        )


    def forward(self, x: torch.Tensor, **kwargs):
        x = torch.view_as_real(x)
        # x = x.transpose(-2,-1).flatten(1,2).contiguous()

        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = torch.cat([y1, y2], dim=-3) # cat along channel dim
        y3 = self.conv3(y3)
        y4 = torch.cat([y1, y2, y3], dim=-3) # cat along channel dim
        y = self.conv4(y4)

        y = self.mlp(y)
        return y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)
