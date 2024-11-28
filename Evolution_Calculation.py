# %%
def compute_unitary_evolution(Omega, f1, f2, A, B, T, N=1000):
    """
    Compute the unitary evolution U(T) given Omega, f1(t), f2(t), error parameters A and B, and duration T.

    Parameters:
    Omega: float
        The coupling constant.
    f1: function
        Function of time t, returns the real part of the driving.
    f2: function
        Function of time t, returns the imaginary part of the driving.
    A: float
        Error average
    B: float
        Error difference
    T: float
        Total duration of the evolution.
    N: int
        Number of time steps for the evolution.

    Returns:
    U_T: Qobj
        The unitary evolution operator at time T.
    """
    # Time discretization
    t_list = np.linspace(0, T, N)

    # Single-qubit Pauli operators
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    si = qeye(2)

    # Two-qubit Pauli operators
    sx1 = tensor(sx, si)
    sy1 = tensor(sy, si)
    sx2 = tensor(si, sx)
    sy2 = tensor(si, sy)

    # Define Sx and Sy based on error parameters
    Sx = np.sin(A) * sx1 + np.sin(B) * sx2
    Sy = np.cos(A) * sy1 + np.cos(B) * sy2

    # Hamiltonian terms
    H1 = Omega * Sx
    H2 = Omega * Sy

    # Define time-dependent functions for f1 and f2
    def f1_t(t, args):
        return f1(t)

    def f2_t(t, args):
        return f2(t)

    # Hamiltonian as a list of [Hi, fi]
    H = [[H1, f1_t], [H2, f2_t]]

    args = {}  # Additional arguments (if any)

    # Compute the unitary evolution operator U(T)
    U_T = propagator(H, T, c_ops=[], args=args)

    return U_T


# Usage Example
# Define the piecewise functions f1(t) and f2(t)
def f1(t):
    # Example: f1(t) = 1 for t < T/2, 0 otherwise
    return 1.0 if t < T / 2 else 0.0


def f2(t):
    # Example: f2(t) = 0 for all t
    return 0.0


# Set parameters
Omega = 1.0  # Example value
A = np.pi / 4  # Error parameter for spin 1
B = np.pi / 6  # Error parameter for spin 2
T = 1.0  # Total duration
N = 1000  # Number of time steps

# Compute U(T)
U_T = compute_unitary_evolution(Omega, f1, f2, A, B, T, N)

print("Unitary evolution operator U(T):")
print(U_T)

# %%
