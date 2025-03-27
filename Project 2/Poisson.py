import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def shape_functions_ref(xi):
    """P2 shape functions on reference element [0,1] at coordinate xi."""
    phi1 = 2 * (xi - 0.5) * (xi - 1.0)
    phi2 = 4 * xi * (1.0 - xi)
    phi3 = 2 * xi * (xi - 0.5)
    return np.array([phi1, phi2, phi3])

def shape_functions_ref_deriv(xi):
    """Derivatives of P2 shape functions on reference element [0,1]."""
    dphi1 = 4 * xi - 3.0
    dphi2 = -8 * xi + 4.0
    dphi3 = 4 * xi - 1.0
    return np.array([dphi1, dphi2, dphi3])

def assemble_matrices_and_load(nodes, f_func):
    """Assemble stiffness matrix K, mass matrix M, and load vector F for -u'' = f."""
    N_end = len(nodes)
    n_elem = N_end - 1
    N_dof = 2 * n_elem + 1

    K = lil_matrix((N_dof, N_dof))
    M = lil_matrix((N_dof, N_dof))
    F = np.zeros(N_dof)

    xi_points = [0.0, 0.5, 1.0]
    weights = [1.0, 4.0, 1.0]

    phi_at = {xi: shape_functions_ref(xi) for xi in xi_points}
    dphi_at = {xi: shape_functions_ref_deriv(xi) for xi in xi_points}

    for k in range(n_elem):
        a = nodes[k]
        b = nodes[k+1]
        h = b - a
        i_left = 2*k
        i_mid = 2*k + 1
        i_right = 2*(k+1)
        elem_indices = [i_left, i_mid, i_right]
        local_K = np.zeros((3, 3))
        local_M = np.zeros((3, 3))
        local_F = np.zeros(3)

        for w, xi in zip(weights, xi_points):
            x = a + h * xi
            phi = phi_at[xi]
            dphi = dphi_at[xi]
            fval = f_func(x)
            invL = 1.0 / h
            for i in range(3):
                for j in range(3):
                    local_K[i, j] += w * (dphi[i]*invL) * (dphi[j]*invL) * h
                    local_M[i, j] += w * phi[i] * phi[j] * h
                local_F[i] += w * fval * phi[i] * h

        local_K *= (1.0/6.0)
        local_M *= (1.0/6.0)
        local_F *= (1.0/6.0)

        for local_i, global_i in enumerate(elem_indices):
            for local_j, global_j in enumerate(elem_indices):
                K[global_i, global_j] += local_K[local_i, local_j]
                M[global_i, global_j] += local_M[local_i, local_j]
            F[global_i] += local_F[local_i]

    # Construct full global x_dofs list
    x_dofs = [nodes[0]]
    for k in range(len(nodes) - 1):
        mid = 0.5 * (nodes[k] + nodes[k+1])
        x_dofs.extend([mid, nodes[k+1]])
    x_dofs = np.array(x_dofs)

    return K.tocsr(), M.tocsr(), F, x_dofs

def solve_poisson(nodes, f_func, exact_solution=None):
    """Solve -u'' = f on (0,1) with u(0)=u(1)=0 using P2 FEM."""
    K, M, F, x_dofs = assemble_matrices_and_load(nodes, f_func)
    N_dof = K.shape[0]

    # Dirichlet boundary conditions
    interior_idx = np.arange(1, N_dof - 1)
    K_int = K[interior_idx[:, None], interior_idx]
    F_int = F[interior_idx]

    # Solve system
    u_int = spsolve(K_int, F_int)
    u_full = np.zeros(N_dof)
    u_full[1:N_dof-1] = u_int

    # Error computation
    err_L2 = None
    err_H1 = None

    if exact_solution is not None:
        err_L2_sq = 0.0
        err_H1_sq = 0.0
        n_elem = len(nodes) - 1

        for k in range(n_elem):
            a = nodes[k]
            b = nodes[k+1]
            L = b - a
            mid = 0.5 * (a + b)
            i_left = 2*k
            i_mid = 2*k+1
            i_right = 2*(k+1)
            u_left = u_full[i_left]
            u_mid = u_full[i_mid]
            u_right = u_full[i_right]

            y_left = exact_solution(a)
            y_mid = exact_solution(mid)
            y_right = exact_solution(b)

            h = 1e-6
            dy_left = (exact_solution(a + h) - exact_solution(a)) / h
            dy_mid  = (exact_solution(mid + h) - exact_solution(mid)) / h
            dy_right= (exact_solution(b + h) - exact_solution(b)) / h

            dphi_left  = shape_functions_ref_deriv(0.0)
            dphi_mid   = shape_functions_ref_deriv(0.5)
            dphi_right = shape_functions_ref_deriv(1.0)

            uh_d_left  = (1.0/L) * (u_left*dphi_left[0] + u_mid*dphi_left[1] + u_right*dphi_left[2])
            uh_d_mid   = (1.0/L) * (u_left*dphi_mid[0]  + u_mid*dphi_mid[1]  + u_right*dphi_mid[2])
            uh_d_right = (1.0/L) * (u_left*dphi_right[0]+ u_mid*dphi_right[1]+ u_right*dphi_right[2])

            e_left  = y_left - u_left
            e_mid   = y_mid  - u_mid
            e_right = y_right - u_right
            err_L2_sq += (L/6.0)*(e_left**2 + 4*e_mid**2 + e_right**2)

            ed_left  = dy_left - uh_d_left
            ed_mid   = dy_mid  - uh_d_mid
            ed_right = dy_right - uh_d_right
            err_H1_sq += (L/6.0)*(ed_left**2 + 4*ed_mid**2 + ed_right**2)

        err_L2 = np.sqrt(err_L2_sq)
        err_H1 = np.sqrt(err_H1_sq)

    return u_full, err_L2, err_H1, x_dofs

