"""
Optimal Control Problem Solver (Poisson equation constraint, 1D P2 FEM).
"""
import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve

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
    M = len(nodes) - 1            # Number of elements
    N_dof = 2 * M + 1             # Total P2 degrees of freedom
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

    # Apply homogeneous Dirichlet BCs: remove first and last DoFs
    B_reduced = B_global[1:-1, 1:-1]
    F_reduced = F_global[1:-1, 1:-1]
    
    return B_reduced, F_reduced, N_dof

A,_,_ = assemble_global_matrices([0,0.2,0.4,0.6,0.8,1])
# Set NumPy print options for cleaner display
np.set_printoptions(precision=3, suppress=True, linewidth=120)

print("Reduced stiffness matrix B (with Dirichlet BCs applied):")
print(A)
print("Shape:", A.shape)