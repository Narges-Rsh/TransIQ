import torch
import torch.nn as nn
from .ModelBase import ModelBase
from typing import List
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

#A Transformer model based on raw IQ_samples (with cls_token,and possitional encoding)

class TransDirect(ModelBase):
    
    
    def __init__(
        self,
        input_samples : int,  
        input_channels: int,
        classes: List[str],
        learning_rate: float = 0.0001,
        num_layers=4,
        nhead=2,   
        dim_feedforward=64,   
        dropout_rate=0.1,
        seq_length=128,  
        token_multi = 64,   # number of tokens= seq_length /token_multi              
        **kwargs):
  
        super().__init__(classes=classes, **kwargs)   

        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate
        self.d_model = input_channels*2*token_multi  
        self.num_layers = num_layers
        self.num_classes = len(classes)

        if seq_length % token_multi != 0:
          raise ValueError("Token Multiplier is not divisible by seq length")

        self.token_multi = token_multi  

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model)) 

        #positional encoding layer
        self.positional_encoding = self.build_positional_encoding(seq_length, self.d_model)  

        # Transformer encoder layer
        self.encoder_layer = TransformerEncoderLayer(self.d_model, nhead, dim_feedforward, dropout=dropout_rate,batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        
        #classification layers
        self.fc = self.build_fc(self.num_classes)
        

        self.example_input_array = torch.zeros((1,input_channels,input_samples), dtype=torch.cfloat)  


    def Tokenization(self,signal):     
        signal=torch.view_as_real(signal)      
        reshaped = signal.transpose(1, 2).reshape(signal.shape[0], signal.shape[2], -1)     
        tokens = reshaped.reshape(reshaped.shape[0], -1, reshaped.shape[2] * self.token_multi)   
        cls_tokens = self.cls_token.expand(tokens.size(0), -1,-1)  #(1,1,16)
        tokens = torch.cat((cls_tokens, tokens), dim=1)     #(1,129,16)
        return tokens   #(batch, sequence_lenght, feature_dimension)
   

    def build_positional_encoding(self, seq_lenght, d_model):
         position_enc = torch.zeros(1,seq_lenght, d_model)  
         position = torch.arange(0, seq_lenght).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
         position_enc[:,:, 0::2] = torch.sin(position * div_term)   
         position_enc[:,:, 1::2] = torch.cos(position * div_term)   
            
         return nn.Parameter(position_enc, requires_grad=False)     
    
   
    def build_fc(self,num_classes):
        return nn.Sequential(   
            nn.Flatten(),
            nn.LazyLinear(32),       
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LazyLinear(num_classes),    
        )
                                                                    

    def forward(self, x: torch.Tensor, **kwargs):
        x=self.Tokenization(x)  
          
        #add positional encoding
        x += self.positional_encoding[:, :x.size(1), :]    
        
        #Transformer encoder
        x = self.transformer_encoder(x)     

        # extract Cls Token   
        x = x[:, 0, :]

        # Fully connected 
        x = self.fc(x)

        return x   

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)  
    


