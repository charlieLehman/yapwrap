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
        num_classes = input.size(1)
        c_out = torch.zeros_like(input)
        for k in range(num_classes):
            _out = torch.cat([input[:,:k] , input[:,(k+1):]],1)
            c_out[:,k] = -torch.logsumexp(_out, 1)
        return c_out

    def forward(self, input):
        # Extract the negative max complement logits
        out = self.model(input)
        if self.training:
            return self.cc(out)
        else:
            return out

    def visualize(self, x):
        mhp = HistPlot(title='Model Logit Response',
                                   xlabel='Logit',
                                   ylabel='Frequency',
                                   legend=True,
                                   legend_pos=1,
                                   grid=True)

        mmhp = HistPlot(title='Model Max Logit Response',
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

        mcchp = HistPlot(title='Complement Constraint Max Logit Response',
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

        _mout = mout.detach().cpu().numpy()
        mmout = _mout.max(1)
        amout = _mout.argmax(1)
        for n in range(mout.size(1)):
            _x = _mout[:,n]
            mhp.add_plot(_x, label=n)
        mmhp.add_plot(mmout)
        viz_dict.update({'LogitResponse':torch.from_numpy(mhp.get_image()).permute(2,0,1)})
        viz_dict.update({'MaxLogitResponse':torch.from_numpy(mmhp.get_image()).permute(2,0,1)})
        _cout = cout.detach().cpu().numpy()
        mcout = _cout.max(1)
        acout = _cout.argmax(1)
        for n in range(cout.size(1)):
            _x = _cout[:,n]
            cchp.add_plot(_x, label=n)
        mcchp.add_plot(mcout)
        viz_dict.update({'CompConstLogitResponse':torch.from_numpy(cchp.get_image()).permute(2,0,1)})
        viz_dict.update({'CompConstMaxLogitResponse':torch.from_numpy(mcchp.get_image()).permute(2,0,1)})
        mhp.close()
        mmhp.close()
        cchp.close()
        mcchp.close()
        return viz_dict


    @property
    def default_optimizer(self):
        if hasattr(self.model, "default_optimizer"):
            return self.model.default_optimizer
        else:
            return None

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

