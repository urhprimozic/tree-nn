import torch 
import torch.nn as nn 
from collections.abc import Sequence
from torch.distributions import Categorical

class DecisionUnit(nn.Module):
    '''
    TODO

    '''
    def __init__(self, in_features : int, out_paths : int, *args, **kwargs):
        '''
        TODO
        '''
        # init parent
        super().__init__(*args, **kwargs)
        
        # store settings
        self.in_features = in_features
        self.out_paths = out_paths

        # create layers
        self.linear = nn.Linear(in_features, out_paths)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim0 = nn.Softmax(dim=0)

        
    def forward(self, x):
        '''
        TODO
        '''
        x = self.linear(x)
        if x.dim() == 1:
            probs = self.softmax_dim0(x)
        elif x.dim() == 2:
            probs = self.softmax_dim1(x)
        else:
            raise ValueError(f"x.dim() should be 1 or 2, and not {x.dim()} for x = {x}." )
        return torch.reshape(probs, (self.out_paths))
    
class Tree(nn.Module):
    '''
    TODO
    '''
    def __init__(self,in_features : int, out_paths : int, children : Sequence[nn.Module],  *args, **kwargs):
        '''
        TODO

        Creates a new Tree(children) with in_features input features. 

        Parameters
        -----------
        children : Sequence[nn.Module]
            A sequence of different nn modules, used for output. 
        '''
        super().__init__(*args, **kwargs)
        
        self.distribution = DecisionUnit(in_features, out_paths)
        self.children = children
        
    
    def forward(self, x, select_max=True):
        '''
        TODO

        if select_max=True, always take the most probable path TODO
        '''
        # get probabilities 
        probs = self.distribution(x)

        if select_max:
            # return prediction of the most probable model
            index = torch.argmax(probs)
            model = self.children[index]
            return model(x)
        else:
            # return prediction of a model, sampled from categorized(probs)
            dist = Categorical(probs)
            index = dist.sample()
            model = self.children[index]
            return model(x)
            




