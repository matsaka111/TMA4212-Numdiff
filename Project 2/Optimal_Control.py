"""
Optimal Control Problem Solver (Poisson equation constraint, 1D P2 FEM).
"""
import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve

def shape_functions_ref(xi):
    """Reference element shape functions (P2) values at xi."""
    return np.array([
        2 * (xi - 0.5) * (xi - 1.0),    # phi1
        4 * xi * (1.0 - xi),           # phi2
        2 * xi * (xi - 0.5)            # phi3
    ])

def shape_functions_ref_deriv(xi):
    """Reference element shape function derivatives at xi."""
    return np.array([
        4 * xi - 3.0,   # phi1'
        4.0 - 8 * xi,   # phi2'
        4 * xi - 1.0    # phi3'
    ])

def assemble_stiffness_mass(nodes):
    """Assemble global stiffness (K) and mass (M) matrices for P2 elements."""
    N_end = len(nodes)
    n_elem = N_end - 1
    N_dof = 2 * n_elem + 1
    data_K = []; row_K = []; col_K = []
    data_M = []; row_M = []; col_M = []
    xi_pts = [0.0, 0.5, 1.0]; w = [1.0, 4.0, 1.0]
    phi_vals = {xi: shape_functions_ref(xi) for xi in xi_pts}
    dphi_vals = {xi: shape_functions_ref_deriv(xi) for xi in xi_pts}
    for k in range(n_elem):
        a = nodes[k]; b = nodes[k+1]; L = b - a
        i_left = 2*k; i_mid = 2*k + 1; i_right = 2*(k+1)
        for wt, xi in zip(w, xi_pts):
            x = a + L * xi
            phi = phi_vals[xi]; dphi = dphi_vals[xi]
            dx = L; invL = 1.0 / L
            for i_local, global_i in enumerate([i_left, i_mid, i_right]):
                for j_local, global_j in enumerate([i_left, i_mid, i_right]):
                    # Add contributions to stiffness and mass
                    data_K.append(wt * (dphi[i_local]*invL) * (dphi[j_local]*invL) * dx)
                    row_K.append(global_i); col_K.append(global_j)
                    data_M.append(wt * phi[i_local] * phi[j_local] * dx)
                    row_M.append(global_i); col_M.append(global_j)
    # Simpson factor 1/6 for integrals
    data_K = [val/6.0 for val in data_K]
    data_M = [val/6.0 for val in data_M]
    K = csr_matrix((data_K, (row_K, col_K)), shape=(N_dof, N_dof))
    M = csr_matrix((data_M, (row_M, col_M)), shape=(N_dof, N_dof))
    return K, M

def assemble_load_vector(nodes, g_func):
    """Assemble load vector for ∫ g(x)*φ_i(x) dx (using Simpson's rule on each element)."""
    N_end = len(nodes)
    n_elem = N_end - 1
    N_dof = 2 * n_elem + 1
    F = np.zeros(N_dof)
    for k in range(n_elem):
        a = nodes[k]; b = nodes[k+1]; L = b - a; mid = 0.5 * (a + b)
        i_left = 2*k; i_mid = 2*k + 1; i_right = 2*(k+1)
        F[i_left]  += (L/6.0) * g_func(a)
        F[i_mid]   += (2*L/3.0) * g_func(mid)
        F[i_right] += (L/6.0) * g_func(b)
    return F

def solve_optimal_control(nodes, y_d_func, alpha):
    """Solve the optimal control problem for given desired state y_d and cost alpha.
    Returns (Y, P, U) arrays for state, adjoint, and control at all nodes."""
    K, M = assemble_stiffness_mass(nodes)
    N_dof = K.shape[0]
    int_slice = slice(1, N_dof-1)  # interior indices for y and p
    n_int = N_dof - 2             # number of interior y/p unknowns
    n_all = N_dof                 # number of control unknowns (including boundaries)
    # Sub-blocks for saddle-point system
    K_ii = K[int_slice, int_slice]
    M_ii = M[int_slice, int_slice]
    M_int_all = M[int_slice, :]
    M_all_int = M[:, int_slice]
    # Construct block matrix using bmat:
    A = bmat([
        [K_ii, None, -M_int_all],
        [-M_ii, K_ii, None],
        [None, M_all_int, alpha * M]
    ], format='csr')
    # Right-hand side
    F = assemble_load_vector(nodes, y_d_func)  # load vector
    b1 = np.zeros(n_int)          # state eq RHS (0)
    b2 = -F[1:N_dof-1]            # adjoint eq RHS (-F on interior)
    b3 = np.zeros(n_all)          # optimality eq RHS (0)
    b = np.concatenate([b1, b2, b3])
    # Solve for [Y_int, P_int, U] vector
    sol = spsolve(A, b)
    Y_int = sol[0:n_int]
    P_int = sol[n_int:2*n_int]
    U = sol[2*n_int:2*n_int + n_all]
    # Insert boundary values (0) for full state and adjoint
    Y = np.zeros(N_dof); P = np.zeros(N_dof)
    Y[1:N_dof-1] = Y_int
    P[1:N_dof-1] = P_int
    return Y, P, U

def l2_norm(vector, M):
    """Compute L2 norm sqrt(v^T M v) for a function represented by 'vector'. """
    return np.sqrt(vector.dot(M.dot(vector)))

if __name__ == "__main__":
    # Desired state cases
    yd1 = lambda x: 0.5 * x * (1 - x)                        # (1) yd in H1_0, smooth
    yd2 = lambda x: 1.0                                     # (2) yd = 1 (not in H1_0)
    yd3 = lambda x: 1.0 if 0.25 <= x <= 0.75 else 0.0        # (3) piecewise constant (discontinuous)
    cases = [(yd1, "yd = 0.5*x*(1-x)"),
             (yd2, "yd = 1 (constant)"),
             (yd3, "yd = 1 on [0.25,0.75], 0 elsewhere")]
    # Mesh including 0.25 and 0.75 as partition points for case 3
    nodes = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    for (ydf, desc) in cases:
        print(f"\nDesired state: {desc}")
        for alpha in [1e-2, 1e-4, 1e-6, 1e-8]:
            Y, P, U = solve_optimal_control(nodes, ydf, alpha)
            # Compute norms: ||y - y_d||_L2 and ||u||_L2
            Yd_vals = np.array([ydf(x) for x in nodes])
            diff = Y - Yd_vals
            # Use mass matrix for L2 norm calculations
            _, M = assemble_stiffness_mass(nodes)
            err_norm = l2_norm(diff, M)
            u_norm = l2_norm(U, M)
            print(f" alpha={alpha:.0e}: ||y - yd||_L2 = {err_norm:.3e}, ||u||_L2 = {u_norm:.3e}")
        # As alpha increases, control is smaller and state deviates more from yd;
        # as alpha -> 0, control grows and state better approximates yd.
