#!/usr/bin/env python3
import numpy as np
import time
import pickle

class KS_solver():
    def __init__(self, initial_condition, dt, x, model_path, param_path):
        self.u = np.array(initial_condition)
        self.u0 = np.array(initial_condition)
        self.t = 0
        self.dt = dt
        self.dx = x[1] - x[0]
        self.x = x
        self.N = len(x)
        self.k = np.fft.rfftfreq(self.N, d=self.dx) * 2*np.pi 

        # load model
        with open(PARAM_PATH, 'rb') as f: loaded_parameters = pickle.load(f)
        neuralop = PFNO(*loaded_parameters).to("cpu") 
        neuralop.load_state_dict(torch.load(model_path))
        self.model = neuralop

        # linear operator integration
        viscosity = 1 
        hyperviscosity = 1 
        L_hat = viscosity*self.k**2 - hyperviscosity*self.k**4  # linear operator
       
        self.E = np.exp(dt*L_hat)
        self.E2 = np.exp(dt*L_hat/2)
        M = 16  # number of points for contour integral
        r = np.exp(1j*np.pi*(np.arange(1,M+1)-0.5)/M)  # roots of unity
        LR = dt*L_hat[:,None] + r[None,:]
        Q = dt*np.real(np.mean( (np.exp(LR/2)-1)/LR , axis=1))
        self.f1 = dt*np.real(np.mean( (-4 - LR + np.exp(LR)*(4 - 3*LR + LR**2))/LR**3 , axis=1))
        self.f2 = dt*np.real(np.mean( (2 + LR + np.exp(LR)*(-2 + LR))/LR**3 , axis=1))
        self.f3 = dt*np.real(np.mean( (-4 -3*LR - LR**2 + np.exp(LR)*(4 - LR))/LR**3 , axis=1))
        
    def advection(self, uhat):     
        ududx = np.fft.rfft(np.fft.irfft(uhat).real*np.fft.irfft(1j*self.k*uhat).real)
        ududx[self.N//3:-self.N//3] = 0 
        return -ududx
        
    def step(self):
        uhat = np.fft.rfft(self.u)
        k1 = self.advection(uhat)
        k2 = self.advection(self.E2*uhat + 0.5*self.dt*k1)
        k3 = self.advection(self.E2*uhat + 0.5*self.dt*k2)
        k4 = self.advection(self.E*uhat + self.dt*k3)
        
        self.u = np.fft.irfft(self.E*uhat + self.f1*k1 + 2*self.f2*(k2+k3) + self.f3*k4)
        self.t += self.dt
        
        return

Nx = 128
dt = 1E-04
L = 8*np.pi
x = np.linspace(0, L, Nx, endpoint=False)
u0 = np.cos(x) + 0.5*np.sin(x-2) + 0.4*np.sin(x/4) - np.cos(x/2)
#u0 = 0.8*np.cos(x) + 0.5*np.sin(x-1) - 0.2*np.sin(x/2) + np.cos(x/4) # new initial condition

solver = KS_solver(u0, dt, x)

data = np.array([solver.u0])
for i in range(2000000):
    solver.step()
    if np.isnan(solver.u).any(): break
    if i%1000 == 0: 
        data = np.concatenate((data, [solver.u]), axis=0)
        print(f'saving at time t={solver.t}. Mass of solution: {np.mean(solver.u).real}')
with open(f'../data/KuramotoSivashinsky/KS_N{Nx}.pkl', 'wb') as file:
    data = pickle.dump(data, file)
print('done!')  
