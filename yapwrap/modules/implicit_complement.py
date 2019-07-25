import torch

class ImplicitComplementFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=1, keepdim=True):
        _exp = torch.exp(input)
        _den = (1+_exp.sum(dim, keepdim=keepdim))
        ctx.save_for_backward(_exp, _den)
        output = 1/_den
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _exp, _den = ctx.saved_tensors
        grad_input = -_exp.div(_den.pow(2)).mul(grad_output)
        return grad_input, None

class ImplicitComplement(torch.nn.Module):
    def __init__(self, dim, keepdim=True):
        super(ImplicitComplement, self).__init__()
        self.dim = dim
        self.keepdim=keepdim

    def forward(self, input):
        return ImplicitComplementFunction.apply(input, self.dim, self.keepdim)

    def extra_repr(self):
        return 'dim={}, keepdim={}'.format(self.dim, self.keepdim)

class ImplicitAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=1, keepdim=True):
        _exp = torch.exp(input)
        _esum = _exp.sum(dim, keepdim=keepdim)
        _den = (1+_esum)
        ctx.save_for_backward(_exp, _den)
        output = _esum/_den
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _exp, _den = ctx.saved_tensors
        grad_input = _exp.div(_den.pow(2)).mul(grad_output)
        return grad_input, None, None

class ImplicitAttention(torch.nn.Module):
    def __init__(self, dim, keepdim=True):
        super(ImplicitAttention, self).__init__()
        self.dim = dim
        self.keepdim=keepdim

    def forward(self, input):
        return torch.sigmoid(torch.logsumexp(input, self.dim,keepdim=self.keepdim))

    def extra_repr(self):
        return 'dim={}, keepdim={}'.format(self.dim, self.keepdim)

class ImplicitAttentionLoss(torch.nn.Module):
    def __init__(self):
        super(ImplicitAttentionLoss, self).__init__()
    def forward(self, attention, implied_attention):
        attention = attention.clone().detach().requires_grad_(False)
        return (-(attention*torch.log(torch.clamp(implied_attention, 1e-5,1))+(1-attention)*torch.log(torch.clamp((1-implied_attention),1e-5,1)))).mean()
