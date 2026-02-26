import torch.nn as nn
from torch import fft, roll
import torch

class H1Loss(nn.Module):
    def __init__(self, reduction="mean", dim=None):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.dim = dim

    # compute first order derivative
    def diff(self, y):
        if self.dim:
            y = roll(y, shifts=(1,)*len(self.dim), dims=self.dim) - y
        else:
            for i in range(1, len(y.shape)-1):
                y = roll(y, shifts=1, dims=i) - y 
        #for i in self.dim: y *= y.shape[i]
        return y
        
    def forward(self, yin, ytarget):
        loss = (yin - ytarget) ** 2 + (self.diff(yin) - self.diff(ytarget))**2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

class H1Loss_Huber(nn.Module):
    def __init__(self, reduction="mean", dim=None):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.dim = dim

    # compute first order derivative
    def diff(self, y):
        if self.dim:
            y = roll(y, shifts=(1,)*len(self.dim), dims=self.dim) - y
        else:
            for i in range(1, len(y.shape)-1):
                y = roll(y, shifts=1, dims=i) - y 
        #for i in self.dim: y *= y.shape[i]
        return y
        
    def forward(self, yin, ytarget):
        loss = torch.nn.functional.huber_loss(
            yin, 
            ytarget,
        reduction=self.reduction
        ) + torch.nn.functional.huber_loss(
            self.diff(yin),  
            self.diff(ytarget),
            reduction=self.reduction
        )
        return loss

class H1Loss_With_MassConservation(nn.Module):
    def __init__(self, reduction="mean", lam=1):
        super().__init__()
        self.loss_fun = H1Loss(reduction)
        self.reduction = reduction
        self.lam = lam
    def integrate(self, x):
        return torch.sum(x, dim=list(range(1, x.dim()-1)), keepdim=True)
        
    def forward(self, yin, ytarget):
        loss2 = self.loss_fun(yin, ytarget)
        loss = self.lam*torch.abs(self.integrate(yin - ytarget))**2
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:  
            loss = loss
        if loss > loss2: return loss
        else: return loss2
    