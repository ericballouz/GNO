#!/usr/bin/env python3
import numpy as np
import time
import pickle

class KS_solver():
    def __init__(self, initial_condition, dt, x, y):
        self.u = np.array(initial_condition)
        self.u0 = np.array(initial_condition)
        self.t = 0
        self.dt = dt
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.x = x
        self.y = y
        self.Nx = len(x)
        self.Ny = len(y)
        self.kx = np.fft.fftfreq(self.Nx, d=self.dx) * 2*np.pi 
        self.ky = np.fft.rfftfreq(self.Ny, d=self.dy) * 2*np.pi
        Kx, Ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.Kx = Kx
        self.Ky = Ky
        self.k2 = Kx**2 + Ky**2

        # linear operator integration
        viscosity = 1 
        hyperviscosity = 2 
        L_hat = viscosity*self.k2 - hyperviscosity*self.k2**2  # linear operator
        print("max(L) =", L_hat.max())
        print("min(L) =", L_hat.min())
 
        self.E = np.exp(dt*L_hat)
        self.E2 = np.exp(dt*L_hat/2)
        
        M = 16  # number of points for contour integral
        r = np.exp(1j*np.pi*(np.arange(1,M+1)-0.5)/M)  # roots of unity 
        LR = dt*L_hat[..., None] + r
        
        Q = dt*np.real(np.mean( (np.exp(LR/2)-1)/LR , axis=-1))
        self.f1 = dt*np.real(np.mean( (-4 - LR + np.exp(LR)*(4 - 3*LR + LR**2))/LR**3 , axis=-1))
        self.f2 = dt*np.real(np.mean( (2 + LR + np.exp(LR)*(-2 + LR))/LR**3 , axis=-1))
        self.f3 = dt*np.real(np.mean( (-4 -3*LR - LR**2 + np.exp(LR)*(4 - LR))/LR**3 , axis=-1))
        
    def advection(self, uhat):     
        NonLin = -0.5*np.fft.rfft2(np.fft.irfft2(1j*self.Kx*uhat).real**2 + np.fft.irfft2(1j*self.Ky*uhat).real**2)

        kx_cut = np.max(self.kx)*2/3
        ky_cut = np.max(self.ky)*2/3
        
        dealias = (np.abs(self.Kx) < kx_cut) & (np.abs(self.Ky) < ky_cut)
        NonLin *= dealias
        NonLin[0,0] = 0
        return NonLin
        
    def step(self):
        uhat = np.fft.rfft2(self.u)
        k1 = self.advection(uhat)
        k2 = self.advection(self.E2*uhat + 0.5*self.dt*k1)
        k3 = self.advection(self.E2*uhat + 0.5*self.dt*k2)
        k4 = self.advection(self.E*uhat + self.dt*k3)
        
        self.u = np.fft.irfft2(self.E*uhat + self.f1*k1 + 2*self.f2*(k2+k3) + self.f3*k4)
        self.t += self.dt
        
        return

Nx = 64
Ny = 64
dt = 5E-05
Lx = 8*np.pi
Ly = 8*np.pi
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')
u0 = 0.1*(np.sin(Y*X/4)+1)*np.cos(X*Y/4)
solver = KS_solver(u0, dt, x, y)

data = np.array([solver.u0])
print(f'saving at time t={solver.t}. Mass of solution: {np.mean(solver.u0).real}')
for i in range(4000000):
    solver.step()
    if np.isnan(solver.u).any(): break
    if i%2000 == 0: 
        data = np.concatenate((data, [solver.u]), axis=0)
        print(f'saving at time t={solver.t:>0.5f}. Mass of solution: {np.mean(solver.u).real:>0.5f}. Energy of solution: {np.mean(solver.u**2/2).real:>0.5f}')
with open(f'../data/KuramotoSivashinsky/KS2D_Nx{Nx}_Ny{Ny}_8pi.pkl', 'wb') as file:
    data = pickle.dump(data, file)
print('done!')  
