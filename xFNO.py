import torch
import random
from torch import nn
import numpy as np
from layers import *


# here, the depth corresponds to timesteps into the future
class FNO(nn.Module):
    # dimensions: tuple with dimension of each element in batch
    # N2: lifted dimension output
    # k_truncate: tuple with max wavenumbers beyond which to truncate
    # depth: layer depth
    def __init__(self, dimensions, N2, k_truncate, depth=1, activation_fun=nn.functional.relu):
        super().__init__()
        
        self.depth = depth
        self.dimensions = dimensions
        self.Nlifted = N2

        self.initial_layer = nn.Linear(dimensions[-1], N2)  # lifting layer
        self.final_layer = nn.Linear(N2, dimensions[-1]) # decoder
   
        # Fourier layer         
        self.stack = nn.ModuleList()
        for _ in range(self.depth):
            self.stack.append(
                self.fourier_layer(dimensions, N2, k_truncate, activation_fun)
                        )      
            
    def fourier_layer(self, dimensions, N2, k_truncate, activation_fun):
        # choose batch normalization      
        if len(dimensions) == 2:
            batch_layer = nn.BatchNorm1d(dimensions[0])
        elif len(dimensions) == 3:
            batch_layer = nn.BatchNorm2d(dimensions[0])
        elif len(dimensions) == 4:
            batch_layer = nn.BatchNorm3d(dimensions[0])
        return nn.Sequential(nFourierLayer(dimensions, N2, k_truncate), batch_layer, activation(activation_fun))
    
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

# Physical-FNO
# mass conservation is used
class PFNO(nn.Module):
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
        mass=0,
        activation_fun=nn.functional.relu):
        super().__init__()
        
        self.depth = depth
        self.dimensions = dimensions
        self.Nlifted = N2

        # we want a common encoder and decoder
        self.initial_layer = nn.Linear(dimensions[-1], N2)  # lifting layer
        self.final_layer = nn.Linear(N2, dimensions[-1]) # decoder
    
        # Fourier layer         
        self.stack = nn.ModuleList()
        for _ in range(self.depth):
            self.stack.append( 
                nn.Sequential(
                        self.initial_layer,
                        self.conservative_fourier_layer(
                            dimensions, 
                            N2,
                            k_truncate, 
                            activation_fun, 
                            conserve_layer, 
                            conserve_dims,
                            mass
                        ),
                        self.final_layer, 
                        conserve_layer(conserve_dims, mass)
                        ) )      
            
    def conservative_fourier_layer(self, dimensions, N2, k_truncate, activation_fun, conserve_layer, conserve_dims=None, mass=0):
        # choose batch normalization      
        if len(dimensions) == 2:
            batch_layer = nn.BatchNorm1d(dimensions[0])
        elif len(dimensions) == 3:
            batch_layer = nn.BatchNorm2d(dimensions[0])
        elif len(dimensions) == 4:
            batch_layer = nn.BatchNorm3d(dimensions[0])
        return nn.Sequential(
            nFourierLayer(dimensions, N2, k_truncate), 
            batch_layer,  
            conserve_layer(conserve_dims, mass), # apply conservation before mixing scales using nonlinearity!!
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


