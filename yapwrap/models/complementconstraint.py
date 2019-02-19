'''Complement Constraint
'''

import torch
import torch.nn as nn

class ComplementConstraint(nn.Module):
    def __init__(self, model):
        super(ComplementConstraint, self).__init__()
        self.model = model
        self.name = '{}_ComplementConstraint'.format(model.name)

    def forward(self, input):
        # Extract the negative max complement logits
        out = self.model(input)
        num_classes = out.size(1)
        c_out = torch.zeros_like(out)
        for k in range(num_classes):
            _out = torch.cat([out[:,:k] , out[:,(k+1):]],1)
            c_out[:,k] = -torch.logsumexp(_out, 1)

        return c_out

class ComplementConstraintCombined(nn.Module):
    def __init__(self, model):
        super(ComplementConstraintCombined, self).__init__()
        self.model = model
        self.name = '{}_ComplementConstraintCombined'.format(model.name)

    def forward(self, input):
        # Extract the negative max complement logits
        out = self.model(input)
        num_classes = out.size(1)
        c_out = torch.zeros_like(out)
        for k in range(num_classes):
            _out = torch.cat([out[:,:k] , out[:,(k+1):]],1)
            c_out[:,k] = -torch.logsumexp(_out, 1)

        if self.training:
            return torch.logsumexp(torch.stack([c_out,out],0),0)

        # Force the model to update the worst side
        l_out = -torch.logsumexp(torch.stack([-c_out,-out],0),0)
        return l_out
