from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cnn.py
class VTCNN2(ModelBase):
    """Convolutional Neural Network based on the "VT_CNN2" Architecture  

    This network is based off of a network for modulation classification first
    introduced in O'Shea et al and later updated by West/Oshea and Hauser et al
    to have larger filter sizes.

    Modifying the first convolutional layer to not use a bias term is a
    modification made by Bryse Flowers due to the observation of vanishing
    gradients during training when ported to PyTorch (other authors used Keras).

    Including the PowerNormalization inside this network is a simplification
    made by Bryse Flowers so that utilization of DSP blocks in real time for
    data generation does not require knowledge of the normalization used during
    training as that is encapsulated in the network and not in a pre-processing
    stage that must be matched up.

    References
        T. J. O'Shea, J. Corgan, and T. C. Clancy, “Convolutional radio modulation
        recognition networks,” in International Conference on Engineering Applications
        of Neural Networks, pp. 213-226, Springer,2016.

        N. E. West and T. O'Shea, “Deep architectures for modulation recognition,” in
        IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), pp.
        1-6, IEEE, 2017.

        S. C. Hauser, W. C. Headley, and A. J.  Michaels, “Signal detection effects on
        deep neural networks utilizing raw iq for modulation classification,” in
        Military Communications Conference, pp. 121-127, IEEE, 2017.
    """

    def __init__(
        self,
        #input_channels: int,
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

        self.model = nn.Sequential()

        # Batch x 1-channel x IQ x input_samples
        self.model.append(nn.Conv2d(
            in_channels=input_channels,
            out_channels=256,
            kernel_size=(1,7),
            padding=(0,3),
            bias=False,
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.BatchNorm2d(256))
        self.model.append(nn.Conv2d(
            in_channels=256,
            out_channels=80,
            kernel_size=(2,7),
            padding=(0,3),
            bias=True,
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.BatchNorm2d(80))

        # Flatten the input layer down to 1-d
        self.model.append(nn.Flatten())

        # Batch x Features
        #self.model.append(nn.Linear(80 * 1 * input_samples, 256))
        self.model.append(nn.LazyLinear(256))
        self.model.append(nn.ReLU())
        self.model.append(nn.BatchNorm1d(256))

        self.model.append(nn.Linear(256, len(classes)))

    def forward(self, x: torch.Tensor, **kwargs): 
        #print("input shape",x.shape)
        x = torch.view_as_real(x)
        #x = x.transpose(-2,-1).flatten(1,2).contiguous()
        y = self.model(x)
        return y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)