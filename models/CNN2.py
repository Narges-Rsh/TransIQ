from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cnn.py
class CNN2(ModelBase):
    """

    References
        K. Tekbiyik, A. R. Ekti, A. Görçin, G. K. Kurt, and C. Keçeci, 
        “Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels,” 
        in 2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring), May 2020, pp. 1-6. 
        doi: 10.1109/VTC2020-Spring48590.2020.9128408.

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

        self.model = nn.Sequential()

        # Batch x 1-channel x input_samples x IQ 
        self.model.append(nn.Conv2d(
            in_channels=input_channels,
            out_channels=256,
            kernel_size=(7,2),
            padding=(3,1),
            bias=False,
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d((2,1)))
        self.model.append(nn.Dropout2d(0.5))

        self.model.append(nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(7,2),
            padding=(3,1),
            bias=True,
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d((2,1)))
        self.model.append(nn.Dropout2d(0.5))

        self.model.append(nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(7,2),
            padding=(3,1),
            bias=True,
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d((2,1)))
        self.model.append(nn.Dropout2d(0.5))

        self.model.append(nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(7,2),
            padding=(3,1),
            bias=True,
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d((2,1)))
        self.model.append(nn.Dropout2d(0.5))

        # Flatten the input layer down to 1-d
        self.model.append(nn.Flatten())

        # Batch x Features
        self.model.append(nn.LazyLinear(128))
        self.model.append(nn.ReLU())
        self.model.append(nn.Linear(128, len(classes)))

    def forward(self, x: torch.Tensor, **kwargs):
        x = torch.view_as_real(x)
        y = self.model(x)
        return y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)