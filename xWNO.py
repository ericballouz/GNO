import torch
import random
from torch import nn
import numpy as np
import pywt
from layers import *

# Wavelet neural operator
# uses a stationary wavelet transform instead of a fourier transforem
class WNO(nn.Module):
    # dimensions: tuple with dimension of each element in batch
    # N2: lifted dimension output
    # k_truncate: tuple with max wavenumbers beyond which to truncate
    # depth: layer depth
    # wavelet: type of orthogonal wavelet
    def __init__(self, dimensions, N2, k_truncate, depth=1, activation_fun=nn.functional.relu, wavelet='db1'):
        super().__init__()
        
        self.depth = depth
        self.dimensions = dimensions
        self.Nlifted = N2

        self.initial_layer = nn.Linear(dimensions[-1], N2)  # lifting layer
        self.final_layer = nn.Linear(N2, dimensions[-1]) # decoder
   
        # Wavelet layer         
        self.stack = nn.ModuleList()
        for _ in range(self.depth):
            self.stack.append(
                self.wavelet_layer(dimensions, N2, k_truncate, activation_fun, wavelet)
                        )      
            
    def wavelet_layer(self, dimensions, N2, k_truncate, activation_fun, wavelet='db1'):
        # choose batch normalization      
        if len(dimensions) == 2:
            batch_layer = nn.BatchNorm1d(dimensions[0])
        elif len(dimensions) == 3:
            batch_layer = nn.BatchNorm2d(dimensions[0])
        elif len(dimensions) == 4:
            batch_layer = nn.BatchNorm3d(dimensions[0])
        return nn.Sequential(SWTLayer(dimensions, N2, k_truncate, wavelet), batch_layer, activation(activation_fun))
    
    def forward(self, x):
        batch_size = x.shape[0]

        # store outputs sequentially
        output = []
        output.append(self.stack[0](self.initial_layer(x)))
        for i in range(1, self.depth):
            output.append(self.stack[i](output[-1]))
        
        # reshape
        for i in range(self.depth): 
            output[i] = torch.reshape(output[i], (batch_size,) + self.dimensions[:-1] + (1, self.Nlifted)) 
        output = torch.cat(tuple(output), dim=-2)
        
        return self.final_layer(output)


# Physical-WNO
# mass conservation is used
class PWNO(nn.Module):
    # dimensions: tuple with dimension of each element in batch
    # N2: lifted dimension output
    # k_truncate: tuple with max wavenumbers beyond which to truncate
    # depth: layer depth
    # conserve_dims: dimensions along which a conservation law applies
    # conserve_layer: layer that imposes a conservation law
    def __init__(
        self, 
        dimensions, 
        N2, 
        k_truncate, 
        depth=1, 
        conserve_dims=None, 
        conserve_layer=conserveMass, 
        activation_fun=nn.functional.relu, 
        wavelet='db1'
    ):
        super().__init__()
        
        self.depth = depth
        self.dimensions = dimensions
        self.Nlifted = N2

        # we want a common encoder and decoder
        self.initial_layer = nn.Linear(dimensions[-1], N2)  # lifting layer
        self.final_layer = nn.Linear(N2, dimensions[-1]) # decoder
    
        # Wavelet layer         
        self.stack = nn.ModuleList()
        for _ in range(self.depth):
            self.stack.append( 
                nn.Sequential(
                        self.initial_layer,
                        self.conservative_wavelet_layer(
                            dimensions, 
                            N2,
                            k_truncate, 
                            activation_fun, 
                            wavelet,
                            conserve_layer, 
                            conserve_dims=None
                        ),
                        self.final_layer, 
                        conserve_layer(dims=conserve_dims)
                        ) )      
            
    def conservative_wavelet_layer(
                self, 
                dimensions, 
                N2, 
                k_truncate, 
                activation_fun, 
                wavelet, 
                conserve_layer, 
                conserve_dims=None
            ):
        # choose batch normalization      
        if len(dimensions) == 2:
            batch_layer = nn.BatchNorm1d(dimensions[0])
        elif len(dimensions) == 3:
            batch_layer = nn.BatchNorm2d(dimensions[0])
        elif len(dimensions) == 4:
            batch_layer = nn.BatchNorm3d(dimensions[0])
        return nn.Sequential(
            SWTLayer(dimensions, N2, k_truncate, wavelet), 
            batch_layer,  
            conserve_layer(conserve_dims), # apply conservation before mixing scales using nonlinearity!!
            activation(activation_fun)
        ) 
    
    def forward(self, x):
        batch_size = x.shape[0]

        # store outputs sequentially
        output = []
        output.append(self.stack[0](x))
        for i in range(1, self.depth):
            output.append(self.stack[i](output[-1]))
        
        # reshape
        for i in range(self.depth): 
            output[i] = torch.reshape(output[i], (batch_size,) + self.dimensions[:-1] + (1, self.dimensions[-1])) 
        output = torch.cat(tuple(output), dim=-2)
        
        return output