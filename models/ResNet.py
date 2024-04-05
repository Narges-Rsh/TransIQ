from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

class residual_unit(nn.Module):
    def __init__(self):
        super(residual_unit, self).__init__()

        self.conv0 = nn.Conv1d(32, 32, 3, padding='same')
        self.bn0 = nn.BatchNorm1d(32)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(32, 32, 3, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x += identity
        x = self.relu1(x)
        return x

class residual_stack(nn.Module):
    def __init__(self, input_chan):
        super(residual_stack, self).__init__()

        self.stack = nn.Sequential()
        self.stack.append(nn.Conv1d(input_chan, 32, 1, padding='same'))
        self.stack.append(residual_unit())
        self.stack.append(residual_unit())
        self.stack.append(nn.MaxPool1d(2, stride=2))

    def forward(self, x):
        return self.stack(x)
        

class ResNet(ModelBase):
    """

    References
        T. J. O'Shea, T. Roy, and T. C. Clancy, 
        “Over the Air Deep Learning Based Radio Signal Classification,” 
        IEEE Journal on Selected Topics in Signal Processing, vol. 12, no. 1, pp. 168-179, Dec. 2017, 
        doi: 10.1109/jstsp.2018.2797022.

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

        # Build model
        self.model = nn.Sequential(
            residual_stack(2*input_channels),
            residual_stack(32),
            residual_stack(32),
            residual_stack(32),
            residual_stack(32),
            residual_stack(32),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(0.3),
            nn.Linear(128, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(0.3),
            nn.Linear(128, len(classes)),
        )
        
    def forward(self, x, **kwargs):
        x = torch.view_as_real(x)
        x = torch.transpose(x, -1, -2)
        x = torch.flatten(x, -3, -2)
        return self.model(x)
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.00001)
