import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

# 1. System Parameters
omega_qubit = 1.0        # Qubit frequency
delta_omega = 0.05       # Frequency error in the laser
epsilon_amplitude = 0.1  # Amplitude error in the laser

# 2. Total Time and Segments
T = 10.0                 # Total evolution time
num_segments = 4
T_seg = T / num_segments

# 3. Pauli Matrices for Each Qubit
sx1 = tensor(sigmax(), qeye(2))
sy1 = tensor(sigmay(), qeye(2))
sz1 = tensor(sigmaz(), qeye(2))

sx2 = tensor(qeye(2), sigmax())
sy2 = tensor(qeye(2), sigmay())
sz2 = tensor(qeye(2), sigmaz())

# 4. Identity Operator for Two Qubits
I = tensor(qeye(2), qeye(2))  # Corrected identity operator

# 5. Ideal MS Gate Unitary
U_target = (-1j * (np.pi / 4) * sx1 * sx2).expm()

# 6. Static Hamiltonian with Frequency Errors
H0 = 0.5 * omega_qubit * (sz1 + sz2) + 0.5 * delta_omega * (sz1 + sz2)

# 7. Control Field Amplitudes (to be optimized)
def initial_params():
    # Initial guess satisfying the zero net area condition
    Ω0 = 1.0
    return np.array([Ω0, -Ω0, Ω0, -Ω0])

def control_amplitudes(params):
    # Include amplitude error
    return params * (1 + epsilon_amplitude)

# 8. Hamiltonians for Each Segment
def segment_hamiltonians(params):
    H_list = []
    amplitudes = control_amplitudes(params)
    for Ω in amplitudes:
        H_control = Ω * (sx1 + sx2)
        H_total = H0 + H_control
        H_list.append(H_total)
    return H_list

# 9. Compute the Total Evolution Operator
def total_evolution_operator(params):
    H_list = segment_hamiltonians(params)
    U_total = I  # Use the corrected identity operator
    for H in reversed(H_list):  # Reverse order due to right-to-left multiplication
        U = (-1j * H * T_seg).expm()
        # Ensure U and U_total have the same dimensions
        if U.dims != U_total.dims:
            print(f"Dimension of U: {U.dims}")
            print(f"Dimension of U_total: {U_total.dims}")
            raise ValueError("Dimension mismatch between U and U_total.")
        U_total = U * U_total
    return U_total

# 10. Objective Function (Gate Infidelity)
def objective(params):
    U_final = total_evolution_operator(params)
    # Gate fidelity calculation
    fidelity = np.abs((U_target.dag() * U_final).tr() / 4)**2
    infidelity = 1 - fidelity
    return infidelity

# 11. Constraint: Zero Net Area Condition
def zero_net_area(params):
    # Sum of Ω_i * T_seg should be zero
    return np.sum(params) * T_seg

nonlinear_constraint = NonlinearConstraint(zero_net_area, 0.0, 0.0)

# 12. Optimization
params0 = initial_params()

result = minimize(
    objective,
    params0,
    method='SLSQP',
    constraints=[nonlinear_constraint],
    options={'maxiter': 1000}
)

# 13. Check Optimization Success
if not result.success:
    print("Optimization failed:", result.message)

# 14. Extract Optimized Parameters
params_opt = result.x

# 15. Compute Final Gate Fidelity and Infidelity
U_final_opt = total_evolution_operator(params_opt)
fidelity_opt = np.abs((U_target.dag() * U_final_opt).tr() / 4)**2
infidelity_opt = 1 - fidelity_opt

print(f"Optimized Parameters (Ω1, Ω2, Ω3, Ω4): {params_opt}")
print(f"Final Gate Fidelity: {fidelity_opt:.6f}")
print(f"Final Gate Infidelity: {infidelity_opt:.6e}")

# 16. Plot the Optimized Control Field
tlist = np.linspace(0, T, 1000)
control_field_opt = np.zeros_like(tlist)
amplitudes_opt = control_amplitudes(params_opt)

for i, t in enumerate(tlist):
    if 0 <= t < T_seg:
        control_field_opt[i] = amplitudes_opt[0]
    elif T_seg <= t < 2*T_seg:
        control_field_opt[i] = amplitudes_opt[1]
    elif 2*T_seg <= t < 3*T_seg:
        control_field_opt[i] = amplitudes_opt[2]
    elif 3*T_seg <= t <= T:
        control_field_opt[i] = amplitudes_opt[3]
    else:
        control_field_opt[i] = 0.0

plt.figure(figsize=(10, 6))
plt.plot(tlist, control_field_opt, label='Optimized Control Field')
plt.xlabel('Time')
plt.ylabel('Control Field Amplitude')
plt.title('Optimized Control Field')
plt.legend()
plt.grid(True)
plt.show()
