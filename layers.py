import torch
from torch import nn
import pywt
import numpy as np

# 1D Fourier Layer
class FourierLayer(nn.Module):
    # M: grid discretizaiton
    # N: output dimension
    # N2: dimension of lifted output
    def __init__(self, M, Nlifted, kmax): 
        super().__init__()
        
        if M <= 0: raise ValueError("in_features must be a positive integer")           
        self.weights = nn.Parameter(torch.rand((Nlifted, Nlifted, M//2+1), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.randn((Nlifted, Nlifted)))
        self.kmax = kmax # truncate wavenumbers

    # Fourier transform, ready input for convolution
    def data_transform(self, x):
        x_transformed = torch.fft.rfft(x, dim=1, norm="ortho")
        x_transformed[:, self.kmax:, :] = 0 # truncate high frequencies
        return x_transformed

    # Transform back into physical space
    def inverse_transform(self, x_transformed):
        return torch.fft.irfft(x_transformed, n=M, dim=1,  norm="ortho")
        
    def forward (self, x):
        # Fourier transform
        x_transformed = self.data_transform(x)

        # Linear convolution
        x_transformed = torch.einsum("bkw, wmk -> bkm", x_transformed, self.weights)

        # Inverse transform + bias 
        x_out = self.inverse_transform(x_transformed) + nn.functional.linear(x, self.bias, bias=None)
        return  x_out 

# n-D Fourier Layer
class nFourierLayer(nn.Module):
    # dimensions: shape of each element of the batch. dimensions[-1] is the dimension of f(x)
    # Nlifted: f(x) is lifted to a Nlifted-dimensional space 
    # kmax: list of indices to zero out for each axis
    def __init__(self, dimensions, Nlifted, kmax, internal_nonlinear=None): 
        super().__init__()

        if len(kmax) != len(dimensions)-1: 
            raise ValueError('Please provide truncation wavenumbers for all dimensions')
        self.dimensions = dimensions 
        self.axes = tuple(range(1, len(dimensions), 1)) # axes start at 1. dim = 0 is the batch dimension
        self.weights = nn.Parameter(torch.rand((Nlifted, Nlifted)+ dimensions[:-2] + (dimensions[-2]//2+1,), dtype=torch.cfloat)) 
        self.bias = nn.Parameter(torch.randn((Nlifted, Nlifted)))      
        
        # indices to truncate
        self.k_to_truncate = tuple(
                                [list(range(kmax[i], dimensions[i]-kmax[i])) for i in range(len(kmax)-1)] +
                                [list(range(kmax[-1], dimensions[-2]//2+1))]
                            )

        # Einstein summation code
        einsum_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        einsum_operand_a = einsum_alphabet[0]
        for i in range(len(dimensions)):
            einsum_operand_a += einsum_alphabet[i+1] # code for operand a  
        einsum_operand_b = einsum_operand_a[-1] + einsum_alphabet[len(dimensions)+1] + einsum_operand_a[1:-1] # code for operand b
        einsum_output = einsum_operand_a[:-1] + einsum_operand_b[1]
        self.einsum_indices = einsum_operand_a + "," + einsum_operand_b + "->" + einsum_output

        self.internal_nonlinear = internal_nonlinear

    # Fourier transform, ready input for convolution
    def data_transform(self, x):
        return torch.fft.rfftn(x, dim=self.axes, norm="ortho")

    # Transform back into physical space
    def inverse_transform(self, x_transformed):
        return torch.fft.irfftn(x_transformed, self.dimensions[:-1], dim=self.axes,  norm="ortho")

    # Truncate
    def truncate(self, x):
        for i, axis in enumerate(self.axes):
            if self.k_to_truncate[i]: 
                x.index_fill_(axis, torch.tensor(self.k_to_truncate[i]), 0)
        return x
        
    def forward (self, x):
        # Apply internal nonlinearity
        if self.internal_nonlinear: 
            x_transformed = self.internal_nonlinear(x)
        
        # Fourier transform
        x_transformed = self.data_transform(x)

        # Truncate
        x_transformed = self.truncate(x_transformed)
        
        # Linear convolution
        x_transformed = torch.einsum(self.einsum_indices, x_transformed, self.weights)

        # Inverse transform + bias 
        x_out = self.inverse_transform(x_transformed) + nn.functional.linear(x, self.bias, bias=None)
        return  x_out 


class SWTLayer(nn.Module):
    # M: grid discretizaiton
    # N: output dimension
    # N2: dimension of lifted output
    def __init__(self, dimensions, Nlifted, kmax, wavelet='db1', level=None, mode="periodization"): 
        super().__init__()
        
        if len(kmax) != len(dimensions)-1: 
            raise ValueError('Please provide truncation wavenumbers for all dimensions') 

        # make sure the dimensions are powers of 2?
        #for M in dimensions[2:-1]:
        #    if bin(M).count('1') != 1: raise ValueError("truncate at a wavelet stage")

        self.dimensions = dimensions 
        self.axes = tuple(range(1, len(dimensions), 1)) # axes start at 1. dim = 0 is the batch dimension

         # wavelet transform specs
        self.wavelet = wavelet
        
        if not level: 
            self.level = []
            self.level += [pywt.dwt_max_level(M, pywt.Wavelet(wavelet).dec_len) for M in dimensions[:-1]]

        for i in range(len(kmax)):
            if kmax[i] > self.level[i]: raise ValueError("pick lower stage to truncate at")  

        # store the indices of the wavelet levels
        self.slices = []
        self.slices += [
            np.arange(0, (self.level[i]+1)*dimensions[i] + dimensions[i], dimensions[i]) for i in range(len(dimensions[:-1]))
        ]

        self.wt_mode = mode # mode for wavelet transform

        self.weights = nn.Parameter(
            torch.rand(
                (Nlifted, Nlifted) + tuple([dimensions[i] * (self.level[i]+1) for i in range(len(dimensions[:-1]))])
                      )
        ) 
        self.bias = nn.Parameter(torch.randn((Nlifted, Nlifted)))

        # indices to truncate
        self.k_to_truncate = tuple([list(range(kmax[i]*self.dimensions[i], dimensions[i]*self.level[i])) for i in range(len(kmax))])
        
        # Einstein summation code
        einsum_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        einsum_operand_a = einsum_alphabet[0]
        for i in range(len(dimensions)):
            einsum_operand_a += einsum_alphabet[i+1] # code for operand a  
        einsum_operand_b = einsum_operand_a[-1] + einsum_alphabet[len(dimensions)+1] + einsum_operand_a[1:-1] # code for operand b
        einsum_output = einsum_operand_a[:-1] + einsum_operand_b[1]
        self.einsum_indices = einsum_operand_a + "," + einsum_operand_b + "->" + einsum_output


    # Fourier transform, ready input for convolution 
    def data_transform(self, x):   
        x_transformed = x.detach().numpy()
        for i in range(len(self.axes)):
            x_transformed = np.concatenate(
                                pywt.swt(
                                    x_transformed, 
                                    self.wavelet, 
                                    level=self.level[i], 
                                    axis=self.axes[i], 
                                    trim_approx=True,
                                    norm=True
                                ), 
                            axis=self.axes[i]) # returns transform as a list of coefficients
        return torch.tensor(x_transformed)

    # Transform back into physical space
    def inverse_transform(self, x_transformed):      
        x = x_transformed.detach().numpy()
        # cast into form compatible with pywt.waverec (list of arrays)
        for i in range(len(self.axes)):
            current_axis = self.axes[i]
            coeffs = [] # coefficients for inverse transform along current_axis 
            skip_slices = [slice(None,)]*current_axis # axes to skip
            for j in range(len(self.slices[i])-1):
                current_slices = skip_slices + [slice(self.slices[i][j], self.slices[i][j+1])]
                coeffs += [x[tuple(current_slices)]]
            x = pywt.iswt(coeffs, wavelet=self.wavelet, axis=current_axis)
        return torch.tensor(x)
        
    # Truncate
    def truncate(self, x):
        for i, axis in enumerate(self.axes):
            if self.k_to_truncate[i]: 
                x.index_fill_(axis, torch.tensor(self.k_to_truncate[i]), 0)
        return x
        
    def forward (self, x):
        # wavelet transform
        x_transformed = self.data_transform(x)

        # Truncate
        x_transformed = self.truncate(x_transformed)
        
        # Linear convolution
        x_transformed = torch.einsum(self.einsum_indices, x_transformed, self.weights)

        # Inverse transform + bias 
        x_out = self.inverse_transform(x_transformed) + nn.functional.linear(x, self.bias, bias=None)
        return  x_out 


# Activation layer
class activation(nn.Module):
    def __init__(self, fun=nn.functional.relu):
        super().__init__()
        self.fun = fun
    def forward(self, x):
        return self.fun(x) 


class conserveMass(nn.Module):
    def __init__(self, dims, mass):
        super().__init__()
        self.dims = dims
        self.mass=mass
        
    def forward(self, x):
        if self.dims:
            x = x - x.mean(dim=self.dims, keepdim=True)
        else: 
            x = x - x.mean(dim=tuple(range(1, x.ndim-1)), keepdim=True) # last dim is the output, first dim is batch size
            #x = x - x.mean(dim=tuple(range(1, x.ndim-2)), keepdim=True) # last dim is the output, first dim is batch size
        if self.mass>0: x+= self.mass
        return x
    

class truncateWavenumbers(nn.Module):
    def __init__(self, kmax, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.axes = tuple(range(1, len(dimensions), 1))
        self.k_to_truncate = tuple(
                                [list(range(kmax[i], dimensions[i]-kmax[i])) for i in range(len(kmax)-1)] +
                                [list(range(kmax[-1], dimensions[-2]//2+1))]
                            )
    def forward(self, x):
        x = torch.fft.rfftn(x, dim=self.axes, norm="ortho")
        for i, axis in enumerate(self.axes):
            if self.k_to_truncate[i]: 
                x.index_fill_(axis, torch.tensor(self.k_to_truncate[i]), 0)
        x = torch.fft.irfftn(x, self.dimensions[:-1], dim=self.axes,  norm="ortho")
        return x



