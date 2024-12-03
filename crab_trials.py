#%%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

T = 10.0 
n_ts = 100
delta_t = T / n_ts

Omega = 1.0 

A = 0.1      
B = 0.05     

sx = sigmax()
sy = sigmay()
sz = sigmaz()
si = qeye(2)

sx1 = tensor(si, sx, si)
sy1 = tensor(si, sy, si)
sz1 = tensor(si,sz, si)
sx2 = tensor(si,si, sx)
sy2 = tensor(si,si, sy)
sz2 = tensor(si,si, sz)



Sx = np.sin(A) * sx1 + np.sin(B) * sx2
Sy = np.cos(A) * sy1 + np.cos(B) * sy2

#How should I define them?

f1 = np.ones(n_ts)
f2 = np.zeros(n_ts)
#combine in one vector
from qutip import qeye, sigmax, sigmay, tensor

def unitary_pw(N, T, Omega, f1, f2):

   delta_t = T / N
   U = tensor(si,si,si)
   X = tensor(sigmax(), si,si)  
   Y = tensor(sigmay(), si,si)

   for k in range(N):
       H_t = Omega * (f1[k] * X + f2[k] * Y)*(Sx + Sy)**2
       U_step = (-1j * H_t * delta_t).expm()
       U = U_step * U

   return U



Sy_ideal = (tensor(si, sy, si) + tensor(si,si,sy))
Z = -0.25 * tensor(si,sz,si)*Sy_ideal**2

theta = np.pi / 2
U_MS = (-1j * theta * Z).expm()

initial_state = tensor(basis(2, 0), basis(2, 0))  # |00⟩

U = unitary_pw(n_ts, T, Omega, f1, f2)
print(U)

#fid = 1/8 * Tr(U*U_MS))^2
#print("Fidelity between U and U_MS:", fid)
# %%