'''Complement Constraint
'''

import torch
import torch.nn as nn
from yapwrap.utils import HistPlot

class ComplementConstraint(nn.Module):
    def __init__(self, model):
        super(ComplementConstraint, self).__init__()
        self.model = model
        self.name = '{}_ComplementConstraint'.format(model.name)

    def cc(self, input):
        num_classes = out.size(1)
        c_out = torch.zeros_like(out)

        for k in range(num_classes):
            _out = torch.cat([out[:,:k] , out[:,(k+1):]],1)
            c_out[:,k] = -torch.logsumexp(_out, 1)
        return c_out

    def forward(self, input):
        # Extract the negative max complement logits
        out = self.model(input)
        return self.cc(out)

    def visualize(self, x):
        mhp = HistPlot(title='Model Logit Response',
                                   xlabel='Logit',
                                   ylabel='Frequency',
                                   legend=True,
                                   legend_pos=1,
                                   grid=True)

        cchp = HistPlot(title='Complement Constraint Logit Response',
                             xlabel='Logit',
                             ylabel='Frequency',
                             legend=True,
                             legend_pos=1,
                             grid=True)
        mout = self.model(x)
        cout = self.cc(mout)
        viz_dict = {}
        if hasattr(self.model, 'visualize'):
            viz = getattr(self.model,'visualize', None)
            if callable(viz):
                viz_dict.update(viz(x))

        for n in range(mout.size(1)):
            _x = mout[:,n].detach().cpu().numpy()
            mhp.add_plot(_x, label=n, rwidth=0.3)
        viz_dict.update({'LogitResponse':torch.from_numpy(mhp.get_image()).permute(2,0,1)})
        for n in range(cout.size(1)):
            _x = cout[:,n].detach().cpu().numpy()
            cchp.add_plot(_x, label=n, rwidth=0.3)
        viz_dict.update({'CompConstLogitResponse':torch.from_numpy(cchp.get_image()).permute(2,0,1)})
        mhp.close()
        cchp.close()
        return viz_dict

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

        return c_out + out

