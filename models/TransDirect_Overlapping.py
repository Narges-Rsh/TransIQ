import torch
import torch.nn as nn
from .ModelBase import ModelBase
from typing import List
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

#A Transformer model using a 50% overlap with the previous token

class TransDirect_Overelapping(ModelBase):
    
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
        d_model=128,        
        **kwargs): 
  
        super().__init__(classes=classes, **kwargs)   

        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate
        self.d_model = d_model 
        self.num_layers = num_layers
        self.num_classes = len(classes)
  
        self.seq_length = seq_length 

        #positional encoding layer
        self.positional_encoding = self.build_positional_encoding(seq_length, self.d_model)  

        # Transformer encoder layer
        self.encoder_layer = TransformerEncoderLayer(self.d_model, nhead, dim_feedforward, dropout=dropout_rate,batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        
        #classification layers
        self.fc = self.build_fc(self.num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model)) 

        self.example_input_array = torch.zeros((1,input_channels,input_samples), dtype=torch.cfloat)  


    def Tokenization(self,signal):      
        signal=torch.view_as_real(signal)      
        reshaped = signal.transpose(1, 2).reshape(signal.shape[0], signal.shape[2], -1)    
      
        
        token_size = 64   
        stride = 32      # overlap of 50%

        # Pad the signal with zeros at the end     
        padding_size=32
        padding = torch.zeros((reshaped.shape[0], padding_size, reshaped.shape[2])).to(reshaped.device)  
        reshaped = torch.cat((reshaped, padding), dim=1)  # Padded signal   
       
        # Tokenization (overlapping)
        tokens = reshaped.unfold(dimension=1, size=token_size, step=stride)  
           
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)     
        cls_tokens = self.cls_token.expand(tokens.size(0), -1,-1)
        tokens = torch.cat((cls_tokens, tokens), dim=1) 

        return tokens  # (batch, num_tokens+1, feature_dimension)      


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
    


