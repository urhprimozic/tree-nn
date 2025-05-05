import torch
import torch.nn as nn

class Shallow(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

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

