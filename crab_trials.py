import numpy as np
from qutip import *
from scipy.optimize import minimize

# Define system parameters
omega0 = 1.0  # Qubit frequency
T = 10.0      # Total evolution time
N = 1000      # Number of time steps
tlist = np.linspace(0, T, N)  # Time array

# Define initial and target states
psi0 = basis(2, 0)       # Initial state |0>
psi_target = basis(2, 1) # Target state |1>

# Define Pauli matrices
sigma_x = sigmax()
sigma_z = sigmaz()

# Define the static Hamiltonian
H0 = 0.5 * omega0 * sigma_z

# Number of basis functions
M = 5

# Generate random frequencies
np.random.seed(42)
omega_k = np.random.uniform(0.5, 2.0, M)

# Define the control field as a sum over basis functions
def control_field(t, coeffs):
    field = 0
    for k in range(M):
        field += coeffs[2*k] * np.sin(omega_k[k] * t) + coeffs[2*k+1] * np.cos(omega_k[k] * t)
    return field

# Objective function to minimize (1 - fidelity)
def objective(coeffs):
    # Time-dependent control Hamiltonian
    Ht = [H0, [sigma_x, lambda t, args: control_field(t, coeffs)]]
    
    # Solve the Schrödinger equation
    result = sesolve(Ht, psi0, tlist)
    psi_final = result.states[-1]
    
    # Compute the fidelity
    fidelity = abs(psi_target.overlap(psi_final))**2
    
    # Return (1 - fidelity) for minimization
    return 1 - fidelity

# Initial guess for coefficients
coeffs0 = np.random.randn(2 * M)

# Perform the optimization
result = minimize(objective, coeffs0, method='Nelder-Mead', options={'maxiter': 500})

# Extract optimized coefficients
coeffs_opt = result.x

# Generate the optimized control field
control_opt = [control_field(t, coeffs_opt) for t in tlist]

# Plot the optimized control field
import matplotlib.pyplot as plt

plt.figure()
plt.plot(tlist, control_opt)
plt.xlabel('Time')
plt.ylabel('Control Field Amplitude')
plt.title('Optimized Control Field using CRAB')
plt.show()

# Simulate the system with the optimized control field
Ht_opt = [H0, [sigma_x, lambda t, args: control_field(t, coeffs_opt)]]
result_opt = sesolve(Ht_opt, psi0, tlist)

# Compute the final fidelity
psi_final_opt = result_opt.states[-1]
fidelity_opt = abs(psi_target.overlap(psi_final_opt))**2
print(f"Final Fidelity: {fidelity_opt:.6f}")

# Plot the population dynamics
pop0 = [abs(state.overlap(basis(2, 0)))**2 for state in result_opt.states]
pop1 = [abs(state.overlap(basis(2, 1)))**2 for state in result_opt.states]

plt.figure()
plt.plot(tlist, pop0, label='|0⟩ Population')
plt.plot(tlist, pop1, label='|1⟩ Population')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Population Dynamics with Optimized Control')
plt.show()
