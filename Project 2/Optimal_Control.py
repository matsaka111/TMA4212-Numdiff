"""
Optimal Control Problem Solver (Poisson equation constraint, 1D P2 FEM).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

def shape_functions_P2(xi):
    psi0 = 2 * (xi - 0.5) * (xi - 1)
    psi1 = 4 * xi * (1 - xi)
    psi2 = 2 * xi * (xi - 0.5)
    return np.array([psi0, psi1, psi2])

def shape_function_derivatives_P2(xi):
    return np.array([4 * xi - 3, 4 - 8 * xi, 4 * xi - 1])

def local_to_global_dof_index(k):
    return [2 * k, 2 * k + 1, 2 * k + 2]

def assemble_local_matrices(a, b):
    h = b - a
    B_local = (1 / (3 * h)) * np.array([
        [7, -8, 1],
        [-8, 16, -8],
        [1, -8, 7]
    ])
    F_local = (h / 30) * np.array([
        [4, 2, -1],
        [2, 16, 2],
        [-1, 2, 4]
    ])
    return B_local, F_local

def assemble_global_matrices(nodes):
    M = len(nodes) - 1
    N_dof = 2 * M + 1
    B_global = np.zeros((N_dof, N_dof))
    F_global = np.zeros((N_dof, N_dof))

    for k in range(M):
        a = nodes[k]
        b = nodes[k + 1]
        B_local, F_local = assemble_local_matrices(a, b)
        global_dofs = local_to_global_dof_index(k)

        for i_local, i_global in enumerate(global_dofs):
            for j_local, j_global in enumerate(global_dofs):
                B_global[i_global, j_global] += B_local[i_local, j_local]
                F_global[i_global, j_global] += F_local[i_local, j_local]

    B_reduced = B_global[1:-1, 1:-1]
    F_reduced = F_global[1:-1, 1:-1]
    
    return B_reduced, F_reduced, N_dof

def solve_optimal_control_sparse(B, F, y_d, alpha):
    B = csr_matrix(B)
    F = csr_matrix(F)

    F_inv_B = spsolve(F, B)
    system_matrix = F + alpha * B @ F_inv_B
    rhs = F @ y_d
    y = spsolve(system_matrix, rhs)
    u = spsolve(F, B @ y)

    return y, u

def interpolate_y_d(y_d_func, nodes):
    M = len(nodes) - 1
    N_dof = 2 * M + 1
    dofs = np.zeros(N_dof)

    for k in range(M):
        a = nodes[k]
        b = nodes[k + 1]
        x0, x1, x2 = a, (a + b) / 2, b
        global_dofs = local_to_global_dof_index(k)
        dofs[global_dofs[0]] = y_d_func(x0)
        dofs[global_dofs[1]] = y_d_func(x1)
        dofs[global_dofs[2]] = y_d_func(x2)

    return dofs[1:-1]

def compute_H1_error(u_num, x_dofs, exact_sol, exact_derivative):
    error_sq = 0.0
    quad_points = [0.0, 0.5, 1.0]
    weights = [1.0, 4.0, 1.0]
    n_elem = (len(x_dofs) - 1) // 2

    for k in range(n_elem):
        iL, iM, iR = 2 * k, 2 * k + 1, 2 * k + 2
        xL, xR = x_dofs[iL], x_dofs[iR]
        h = xR - xL
        uL, uM, uR = u_num[iL], u_num[iM], u_num[iR]

        for xi, w in zip(quad_points, weights):
            phi = shape_functions_P2(xi)
            dphi = shape_function_derivatives_P2(xi)
            x = xL + xi * h

            u_h = uL * phi[0] + uM * phi[1] + uR * phi[2]
            u_h_prime = (1 / h) * (uL * dphi[0] + uM * dphi[1] + uR * dphi[2])

            u_exact = exact_sol(x)
            du_exact = exact_derivative(x)

            err_val = u_exact - u_h
            err_grad = du_exact - u_h_prime
            error_sq += w * (err_val**2 + err_grad**2) * h

    return np.sqrt(error_sq / 6.0)

# ------------------- MAIN SCRIPT -------------------
# Setup y_d = 1
M = 11
nodes = np.linspace(0, 1, M + 1)
alpha = 1e-7
y_d_func = lambda x: np.where((0.25 <= x) & (x <= 0.75), 1.0, 0.0)
y_d_prime = lambda x: 0.0  

# Assemble and interpolate
B, F, N_dof = assemble_global_matrices(nodes)
y_d_vector = interpolate_y_d(y_d_func, nodes)
y, u = solve_optimal_control_sparse(B, F, y_d_vector, alpha)

# Add BCs to solution
y_full = np.zeros(N_dof)
y_full[1:-1] = y

# Global P2 mesh points
x_global = np.zeros(N_dof)
for k in range(M):
    a = nodes[k]
    b = nodes[k + 1]
    x0, x1, x2 = a, (a + b) / 2, b
    dofs = local_to_global_dof_index(k)
    x_global[dofs[0]] = x0
    x_global[dofs[1]] = x1
    x_global[dofs[2]] = x2

# Compute error
error = compute_H1_error(y_full, x_global, y_d_func, y_d_prime)
print(f"H1 error: {error:.4e}")

# Plot
sorted_indices = np.argsort(x_global)
x_sorted = x_global[sorted_indices]
y_sorted = y_full[sorted_indices]
y_d_exact = y_d_func(x_sorted)

plt.figure(figsize=(8, 5))
plt.plot(x_sorted, y_d_exact * np.ones_like(x_sorted), 'k--', label='$y_d(x) = 1$ (target)')
plt.plot(x_sorted, y_sorted, 'b-', marker='o', label='$y_h(x)$ (optimal state)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(f'Optimal State vs Target (M={M}, $\\alpha$={alpha})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
