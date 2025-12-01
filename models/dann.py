# models/dann.py
import torch
import torch.nn as nn
from torch.autograd import Function

class GRL(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambda_ * grad_out, None

def grad_reverse(x, lambda_):
    return GRL.apply(x, lambda_)

class DomainDiscriminator(nn.Module):
    def __init__(self, d_in, num_domains, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Linear(hidden//2, num_domains)
        )
    def forward(self, z, lambda_):
        return self.net(grad_reverse(z, lambda_))
