import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# System parameters
T = 10.0           # Total evolution time
n_ts = 100         # Number of time slices (N)
delta_t = T / n_ts # Time step size

# Driving strength
Omega = 1.0        # Adjust as needed

# Error parameters
A = 0.1            # Error average (adjust as needed)
B = 0.05           # Error difference average (adjust as needed)

# Single-qubit Pauli operators
sx = sigmax()
sy = sigmay()
sz = sigmaz()
si = qeye(2)

# Two-qubit Pauli operators
sx1 = tensor(sx, si)
sy1 = tensor(sy, si)
sz1 = tensor(sz, si)
sx2 = tensor(si, sx)
sy2 = tensor(si, sy)
sz2 = tensor(si, sz)

# Define Sx and Sy based on error parameters
Sx = np.sin(A) * sx1 + np.sin(B) * sx2
Sy = np.cos(A) * sy1 + np.cos(B) * sy2

#How should I define them?
X = 1
Y = 1

# Control functions f1(t) and f2(t) as lists of amplitudes
# For simplicity, we'll use constant amplitudes; adjust as needed
f1 = np.ones(n_ts)  # Real part of the driving
f2 = np.zeros(n_ts) # Imaginary part of the driving

# Control Hamiltonian H(t) will be constructed using Sx and Sy
# Function to compute the unitary evolution U(T) using piecewise constant Hamiltonians
def unitary_pw(N):
    delta_t = T / N  # Time step size
    U = qeye(4)      # Identity operator for two qubits (dimension 4)
    for k in range(N):
        H_t = Omega * (f1[k]*X+f2[k]*Y)*(Sx + Sy)**2
        U_step = (-1j * H_t * delta_t).expm()
        U = U_step * U  # Multiply in time order
    return U

# Define the target Hamiltonian H_XX for the MS gate
H_XX = sx1 * sx2

# Target unitary (MS gate)
theta = np.pi / 2           # Interaction angle
U_MS = (-1j * theta * H_XX).expm()

# Initial state
initial_state = tensor(basis(2, 0), basis(2, 0))  # |00⟩

# Compute the unitary evolution
U = unitary_pw(n_ts)

# Apply the unitary to the initial state
final_state = U * initial_state

# Compute the fidelity between the implemented unitary and the target unitary
fid = fidelity(U, U_MS)
print("Fidelity between U and U_MS:", fid)


"""
# CRAB method parameters
method_params = {
    'n_basis_func': 5,        # Number of basis functions
    'num_optim_iter': 100,    # Number of optimization iterations
}

# Run the optimization
result = pulseoptim.opt_pulse_crab(
    H_drift,               # Drift Hamiltonian
    H_controls,            # Control Hamiltonians
    initial_state,         # Initial state
    U_MS,                  # Target unitary (MS gate)
    n_ts,                  # Number of time slices
    T,                     # Total evolution time
    amp_lbound=-5.0,       # Lower bound on control amplitudes
    amp_ubound=5.0,        # Upper bound on control amplitudes
    fid_err_targ=1e-4,     # Target fidelity error
    max_iter=500,          # Maximum number of iterations
    method_params=method_params,
    gen_stats=True         # Generate statistics
)

# Extract optimized pulses
optimized_pulses = result.final_amps  # Shape: (n_ts, n_controls)

# Time grid
t_list = np.linspace(0, T, n_ts)

# Plot the optimized control pulses
plt.figure(figsize=(12, 6))
control_labels = ['Global X', 'Global Y', 'Spin-Spin Interaction']
for i in range(len(H_controls)):
    plt.plot(t_list, optimized_pulses[:, i], label=control_labels[i])
plt.xlabel('Time')
plt.ylabel('Control amplitude')
plt.title('Optimized Control Pulses for MS Gate')
plt.legend()
plt.show()

# Final fidelity error
final_fid_err = result.fid_err
print(f'Final Fidelity Error: {final_fid_err}')

# Calculate the fidelity
final_fidelity = 1 - final_fid_err
print(f'Final Fidelity: {final_fidelity}')
"""
