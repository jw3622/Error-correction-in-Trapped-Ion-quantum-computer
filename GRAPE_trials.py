# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.control import pulseoptim
import logging

# Suppress logging output for clarity
logging.getLogger().setLevel(logging.ERROR)

# Define system parameters
# Number of qubits
N = 2

# Pauli matrices
sx = sigmax()
sy = sigmay()
sz = sigmaz()

# Identity operator
I = qeye(2)

# Tensor products for two-qubit operators
# Single-qubit operators acting on qubit 0
sx0 = tensor(sx, I)
sy0 = tensor(sy, I)
sz0 = tensor(sz, I)

# Single-qubit operators acting on qubit 1
sx1 = tensor(I, sx)
sy1 = tensor(I, sy)
sz1 = tensor(I, sz)

# Interaction term (Ising-type interaction)
J = 1.0  # Interaction strength
H_d = J * tensor(sx, sx)  # Drift Hamiltonian includes interaction

# Control Hamiltonians
# Controls acting on each qubit
H_c = [sx0, sy0, sx1, sy1]

# Initial and target unitaries
U_0 = qeye(4)  # Identity operator for two qubits

# Mølmer-Sørensen gate
# The MS gate is U_MS = exp(-i * (theta/2) * σ_x ⊗ σ_x)
theta = np.pi / 2  # For a full entangling gate

U_targ = (-1j * (theta / 2) * tensor(sx, sx)).expm()

# GRAPE optimization parameters
n_ts = 100  # Number of time slices
evo_time = 10  # Total evolution time
fid_err_targ = 1e-4  # Fidelity error target
max_iter = 500  # Maximum number of iterations
max_wall_time = 120  # Maximum wall time (seconds)
amp_lbound = -5.0  # Lower bound on control amplitudes
amp_ubound = 5.0  # Upper bound on control amplitudes

# Run the GRAPE optimization
result = pulseoptim.optimize_pulse(
    H_d,
    H_c,
    U_0,
    U_targ,
    n_ts,
    evo_time,
    fid_err_targ=fid_err_targ,
    max_iter=max_iter,
    max_wall_time=max_wall_time,
    amp_lbound=amp_lbound,
    amp_ubound=amp_ubound,
    fid_params={"phase_option": "PSU"},
    init_pulse_type="LIN",
)

# Extract the optimized pulses
opt_amps = result.final_amps

# Get the final evolved unitary
U_final = result.evo_full_final

# Calculate fidelity
# Fidelity is 1 - fid_err, since fid_err = 1 - fidelity
fidelity = 1 - result.fid_err

# Print results
print(f"Final fidelity error: {result.fid_err}")
print(f"Fidelity: {fidelity}")
print(f"Number of iterations: {result.num_iter}")
print(f"Final evolution unitary:\n{U_final}")

# Plot the optimized control pulses
times = np.linspace(0, evo_time, n_ts)
plt.figure(figsize=(12, 6))
for i in range(len(H_c)):
    plt.step(times, opt_amps[:, i], where="post", label=f"Control Hamiltonian {i+1}")
plt.xlabel("Time")
plt.ylabel("Control amplitude")
plt.title("Optimized Control Pulses")
plt.legend()
plt.show()

# %%
