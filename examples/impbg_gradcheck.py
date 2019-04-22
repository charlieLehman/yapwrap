import torch
from torch.autograd import gradcheck
from yapwrap.modules import *

impbg = ImplicitComplementFunction.apply
input = (torch.randn(10,10,4,4,dtype=torch.double,requires_grad=True), 1)
test = gradcheck(impbg, input, eps=1e-6, atol=1e-4)
print(test)

impattn = ImplicitAttentionFunction.apply
input = (torch.randn(10,10,4,4,dtype=torch.double,requires_grad=True), 1)
test = gradcheck(impattn, input, eps=1e-6, atol=1e-4)
print(test)

imattn = ImplicitAttention(1, True)
imbg = ImplicitComplement(1, True)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    x = torch.randn(10,10,1,1,dtype=torch.double,requires_grad=True)
    y = impattn(x)
    y.sum().backward()

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for _ in range(10000):
        x = torch.randn(1,1000,1,1,dtype=torch.double,requires_grad=True)
        y = impattn(x)
        y.backward()
print('ImplicitAttention: ', prof.total_average())

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for _ in range(10000):
        x = torch.randn(1,1000,1,1,dtype=torch.double,requires_grad=True)
        y = impbg(x)
        y.backward()

print('ImplicitComplement: ', prof.total_average())





