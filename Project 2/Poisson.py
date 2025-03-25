"""
Poisson Equation FEM Solver (1D, P2 elements).
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def shape_functions_ref(xi):
    """Shape functions on reference element [0,1] at coordinate xi."""
    # P2 Lagrange basis: φ1(0)=1, φ1(0.5)=0, φ1(1)=0; φ2(0)=0, φ2(0.5)=1, φ2(1)=0; φ3(0)=0, φ3(0.5)=0, φ3(1)=1.
    phi1 = 2 * (xi - 0.5) * (xi - 1.0)
    phi2 = 4 * xi * (1.0 - xi)
    phi3 = 2 * xi * (xi - 0.5)
    return np.array([phi1, phi2, phi3])

def shape_functions_ref_deriv(xi):
    """Derivatives of shape functions on reference element [0,1] at coordinate xi."""
    dphi1 = 4 * xi - 3.0    # derivative of φ1
    dphi2 = 4.0 - 8 * xi    # derivative of φ2
    dphi3 = 4 * xi - 1.0    # derivative of φ3
    return np.array([dphi1, dphi2, dphi3])

def assemble_matrices_and_load(nodes, f_func):
    """Assemble global stiffness matrix, mass matrix, and load vector for -u'' = f."""
    N_end = len(nodes)
    n_elem = N_end - 1
    N_dof = 2 * n_elem + 1
    K = lil_matrix((N_dof, N_dof))
    M = lil_matrix((N_dof, N_dof))
    F = np.zeros(N_dof)
    # Simpson's rule (3-point) on each element:
    xi_points = [0.0, 0.5, 1.0]
    weights = [1.0, 4.0, 1.0]
    # Precompute shape values on reference element
    phi_at = {xi: shape_functions_ref(xi) for xi in xi_points}
    dphi_at = {xi: shape_functions_ref_deriv(xi) for xi in xi_points}
    for k in range(n_elem):
        a = nodes[k]; b = nodes[k+1]; L = b - a
        i_left = 2*k; i_mid = 2*k + 1; i_right = 2*(k+1)
        elem_indices = [i_left, i_mid, i_right]
        local_K = np.zeros((3, 3))
        local_M = np.zeros((3, 3))
        local_F = np.zeros(3)
        for w, xi in zip(weights, xi_points):
            x = a + L * xi
            phi = phi_at[xi]
            dphi = dphi_at[xi]
            fval = f_func(x)
            dx = L; invL = 1.0 / L
            for i in range(3):
                for j in range(3):
                    local_K[i, j] += w * (dphi[i]*invL) * (dphi[j]*invL) * dx
                    local_M[i, j] += w * phi[i] * phi[j] * dx
                local_F[i] += w * fval * phi[i] * dx
        local_K *= (1.0/6.0)
        local_M *= (1.0/6.0)
        local_F *= (1.0/6.0)
        for local_i, global_i in enumerate(elem_indices):
            for local_j, global_j in enumerate(elem_indices):
                K[global_i, global_j] += local_K[local_i, local_j]
                M[global_i, global_j] += local_M[local_i, local_j]
            F[global_i] += local_F[local_i]

        x_dofs = [nodes[0]]
        for k in range(len(nodes) - 1):
            xL = nodes[k]
            xR = nodes[k+1]
            mid = 0.5 * (xL + xR)
            x_dofs.extend([mid, xR])
        x_dofs = np.array(x_dofs)
    return K.tocsr(), M.tocsr(), F

def solve_poisson(nodes, f_func, exact_solution=None):
    """Solve -u'' = f on (0,1) with u(0)=u(1)=0 using P2 elements. 
    Returns (solution, L2_error, H1_error)."""
    K, M, F = assemble_matrices_and_load(nodes, f_func)
    N_dof = K.shape[0]
    # Apply Dirichlet BC
    interior_idx = np.arange(1, N_dof-1)
    K_int = K[interior_idx[:, None], interior_idx]
    F_int = F[interior_idx]
    u_int = spsolve(K_int, F_int)
    u_full = np.zeros(N_dof)
    u_full[1:N_dof-1] = u_int
    err_L2 = None; err_H1 = None
    if exact_solution is not None:
        # Calculate errors via Simpson's rule on each element
        err_L2_sq = 0.0; err_H1_sq = 0.0
        n_elem = len(nodes) - 1
        for k in range(n_elem):
            a = nodes[k]; b = nodes[k+1]; L = b - a; mid = 0.5*(a + b)
            i_left = 2*k; i_mid = 2*k+1; i_right = 2*(k+1)
            u_left = u_full[i_left]; u_mid = u_full[i_mid]; u_right = u_full[i_right]
            # exact solution and its derivative at quadrature points
            y_left = exact_solution(a); y_mid = exact_solution(mid); y_right = exact_solution(b)
            # (If exact derivative formula known, use it; here using finite diff for simplicity)
            h = 1e-6
            dy_left = (exact_solution(a+h) - exact_solution(a)) / h
            dy_mid = (exact_solution(mid+h) - exact_solution(mid)) / h
            dy_right = (exact_solution(b+h) - exact_solution(b)) / h
            # FE solution derivative at quadrature points
            dphi_left = shape_functions_ref_deriv(0.0)
            dphi_mid  = shape_functions_ref_deriv(0.5)
            dphi_right= shape_functions_ref_deriv(1.0)
            uh_d_left  = (1.0/L)*(u_left*dphi_left[0] + u_mid*dphi_left[1] + u_right*dphi_left[2])
            uh_d_mid   = (1.0/L)*(u_left*dphi_mid[0]  + u_mid*dphi_mid[1]  + u_right*dphi_mid[2])
            uh_d_right = (1.0/L)*(u_left*dphi_right[0]+ u_mid*dphi_right[1]+ u_right*dphi_right[2])
            # Simpson's rule on this element for errors
            e_left = y_left - u_left; e_mid = y_mid - u_mid; e_right = y_right - u_right
            err_L2_sq += (L/6.0)*(e_left**2 + 4*e_mid**2 + e_right**2)
            ed_left = dy_left - uh_d_left; ed_mid = dy_mid - uh_d_mid; ed_right = dy_right - uh_d_right
            err_H1_sq += (L/6.0)*(ed_left**2 + 4*ed_mid**2 + ed_right**2)
        err_L2 = np.sqrt(err_L2_sq); err_H1 = np.sqrt(err_H1_sq)
    return u_full, err_L2, err_H1

if __name__ == "__main__":
    # Example test for f(x)=1 with exact solution u(x) = 0.5*x*(1-x)
    f = lambda x: 1.0
    exact_sol = lambda x: 0.5 * x * (1 - x)
    # partition
    nodes = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
    u_num, L2_error, H1_error = solve_poisson(nodes, f, exact_solution=exact_sol)
    print("Computed solution at nodes:", u_num)
    print("L2 error:", L2_error)
    print("H1 error:", H1_error)
