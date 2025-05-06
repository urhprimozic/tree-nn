import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, *dims, dropout=0.1):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dims[-1]:  # Don't add activation/dropout after final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.layers = layers
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, training=False):
        '''
        TODO

        if training is True, a list of tensors will be returned, where each element is output of the next layer
        '''
        if training:
            ans = []
            for layer in self.layers:
                # map thru layer 
                x = layer(x)
                
                ans.append(x)
            return ans
        else:
            return self.net(x)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = nn.Identity()
    def forward(self, x, training=False):
        return self.id(x)
    
Id = Identity()