import torch
import torch.nn as nn
from collections.abc import Sequence
from torch.distributions import Categorical

# TODO incomporate Loss:
# nekak morš passat loss skos tree-je in childrene


class DecisionUnit(nn.Module):
    """
    TODO

    """

    def __init__(self, in_features: int, out_paths: int, *args, **kwargs):
        """
        TODO
        """
        # init parent
        super().__init__(*args, **kwargs)

        # store settings
        self.in_features = in_features
        self.out_paths = out_paths

        # create layers
        self.linear = nn.Linear(in_features, out_paths)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim0 = nn.Softmax(dim=0)

    def n_params(self):
        return self.in_features * self.out_paths + self.out_paths

    def forward(self, x):
        """
        TODO
        """
        x = self.linear(x)
        if x.dim() == 1:
            probs = self.softmax_dim0(x)
        elif x.dim() == 2:
            probs = self.softmax_dim1(x)
        else:
            raise ValueError(
                f"x.dim() should be 1 or 2, and not {x.dim()} for x = {x}."
            )
        return probs
      #  return torch.reshape(probs, (self.out_paths,))


class TreeModelContainer(nn.Module):
    """
    Add expected_value method for a model
    """

    def __init__(self, model: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.is_tree = model.type == Tree

    def forward(self, x, training=False):
        return self.model(x, training=training)

    def expected_value(self, f, x, training=False):
        """
        If self.model is a Tree, returns E(f(path(x)))
        else returns f(model(x))

        """
        if self.is_tree:
            return self.model.expected_value(f, x, training=training)
        else:
            return f(self.model(x, training=training))


class Tree(nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        in_features: int,
        children: Sequence[nn.Module],
        head = None,
        *args,
        **kwargs,
    ):
        """
        TODO

        Creates a new Tree(children) with in_features input features.

        Parameters
        -----------
        children : Sequence[nn.Module]
            A sequence of different nn modules, used for output.

        head : nn.Module
            A model, which transforms the input before it gets passed thrue a tree.
        """
        super().__init__(*args, **kwargs)

        out_paths = len(children)

        self.distribution = DecisionUnit(in_features, out_paths)
        # puts children in containers
        self.branches = nn.ModuleList([TreeModelContainer(child) for child in children])
        self.head = head

    def n_params(self):
        return sum([p.nelement()   for p in self.parameters()  ])

    def forward(self, x, select_max=True, training=False):
        """
        TODO
        Samples a path, and passes input thru the sampled path.

        if select_max=True, always take the most probable path TODO
        """

        # if head exists, use it
        if self.head is not None:
            x = self.head(x)
        # get probabilities
        probs = self.distribution(x)

        if select_max:
            # return prediction of the most probable model
            index = int(torch.argmax(probs).item())
            model = self.branches[index]
            return model(x, training=training)
        else:
            # return prediction of a model, sampled from categorized(probs)
            dist = Categorical(probs)
            index = dist.sample().item()
            model = self.branches[index]
            return model(x, training=training)

    def expected_value(self, f, x, eps=None, training=False):
        """
        Returns E(f(Path(x))). If eps is not none, paths with probability < eps are skipped
        """
        # if input is unbatched, batch it first 
        unbatched=False
        if x.dim() == 1:
            unbatched = True 
            x = torch.reshape(x, (1, x.shape[0]))

      
        # if head exists, use it
        if self.head is not None:
            x = self.head(x)


        if eps is None:
            eps = 0

        #########################################################
        # return E(f(Path(x))) = \sum _{path} P(path) f(Path(x))
        #########################################################

        # map input over all the branches 
        branches = [model.expected_value(f, x, training=training) for model in self.branches]
        branches = torch.stack(branches)
        # get probabilities of different branches 
        probs = self.distribution(x)
        # transpose 
        probs_t = probs.transpose(0,1)
        # expected value is weighted sum of different branches, weighted by probabilities
        E = ( branches * probs_t.unsqueeze(2) ).sum(dim=0)
        
        if unbatched:
            E = torch.reshape(E, E[0].shape)
            return E
        return E
        
        
