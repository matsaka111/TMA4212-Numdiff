import numpy as np
import matplotlib.pyplot as plt

def create_partition(a, b, M, method="uniform"):
    if method == "uniform":
        return np.linspace(a, b, M + 1)
    elif method == "squared":
        return np.linspace(a, b, M + 1) ** 2
    elif method == "random":
        interior = np.sort(np.random.rand(M - 1))
        return np.concatenate(([a], interior, [b]))
    else:
        raise ValueError("Unknown method. Choose 'uniform', 'squared', or 'random'.")

def shape_functions_P2(xi):
    psi0 = 2 * (xi - 0.5) * (xi - 1)
    psi1 = 4 * xi * (1 - xi)
    psi2 = 2 * xi * (xi - 0.5)
    return np.array([psi0, psi1, psi2])

def shape_function_derivatives_P2(xi):
    return np.array([4 * xi - 3, 4 - 8 * xi, 4 * xi - 1])

def reference_to_physical(xi, a, b):
    return a + (b - a) * xi

def local_to_global_dof_index(k):
    return [2 * k, 2 * k + 1, 2 * k + 2]

def assemble_local_element_matrices(a, b, f):
    A_local = np.zeros((3, 3))
    b_local = np.zeros(3)
    quad_points = [0.0, 0.5, 1.0]
    quad_weights = [1.0, 4.0, 1.0]
    h = b-a
    for xi, w in zip(quad_points, quad_weights):
        phi = shape_functions_P2(xi)
        dphi = shape_function_derivatives_P2(xi)
        x = reference_to_physical(xi, a, b)
        f_val = f(x)
        for i in range(3):
            for j in range(3):
                A_local[i, j] += w * dphi[i] * dphi[j] * (1/h)**2 * h
            b_local[i] += w * f_val * phi[i] * h
    A_local *= 1.0 / 6.0
    b_local *= 1.0 / 6.0
    return A_local, b_local

def assemble_global_system(nodes, f):
    M = len(nodes) - 1
    N_dof = 2 * M + 1
    A_global = np.zeros((N_dof, N_dof))
    b_global = np.zeros(N_dof)
    for k in range(M):
        a = nodes[k]
        b = nodes[k + 1]
        A_local, b_local = assemble_local_element_matrices(a, b, f)
        global_dofs = local_to_global_dof_index(k)
        for i_local, i_global in enumerate(global_dofs):
            b_global[i_global] += b_local[i_local]
            for j_local, j_global in enumerate(global_dofs):
                A_global[i_global, j_global] += A_local[i_local, j_local]
    A_reduced = A_global[1:-1, 1:-1]
    b_reduced = b_global[1:-1]
    return A_reduced, b_reduced, N_dof

def solve_poisson_fem(nodes, f):
    A_reduced, b_reduced, N_dof = assemble_global_system(nodes, f)
    u_reduced = np.linalg.solve(A_reduced, b_reduced)
    u_full = np.zeros(N_dof)
    u_full[1:-1] = u_reduced
    x_dofs = [nodes[0]]
    for k in range(len(nodes) - 1):
        xL = nodes[k]
        xR = nodes[k + 1]
        mid = 0.5 * (xL + xR)
        x_dofs.extend([mid, xR])
    x_dofs = np.array(x_dofs)
    return u_full, x_dofs

def compute_L2_error(u_num, x_dofs, exact_sol):
    error_sq = 0.0
    quad_points = [0.0, 0.5, 1.0]
    weights = [1.0, 4.0, 1.0]
    n_elem = (len(x_dofs) - 1) // 2
    for k in range(n_elem):
        iL, iM, iR = 2*k, 2*k+1, 2*k+2
        xL, xR = x_dofs[iL], x_dofs[iR]
        h = xR - xL
        uL, uM, uR = u_num[iL], u_num[iM], u_num[iR]
        for xi, w in zip(quad_points, weights):
            phi = shape_functions_P2(xi)
            x = reference_to_physical(xi, xL, xR)
            u_h = uL * phi[0] + uM * phi[1] + uR * phi[2]
            err = exact_sol(x) - u_h
            error_sq += w * err**2 * h
    return np.sqrt(error_sq / 6.0)

def compute_H1_error(u_num, x_dofs, exact_sol):
    error_sq = 0.0
    quad_points = [0.0, 0.5, 1.0]
    weights = [1.0, 4.0, 1.0]
    n_elem = (len(x_dofs) - 1) // 2

    for k in range(n_elem):
        iL, iM, iR = 2*k, 2*k+1, 2*k+2
        xL, xR = x_dofs[iL], x_dofs[iR]
        h = xR - xL
        uL, uM, uR = u_num[iL], u_num[iM], u_num[iR]

        for xi, w in zip(quad_points, weights):
            phi = shape_functions_P2(xi)
            dphi = shape_function_derivatives_P2(xi)
            x = reference_to_physical(xi, xL, xR)

            u_h = uL * phi[0] + uM * phi[1] + uR * phi[2]
            u_h_prime = (1/h) * (uL * dphi[0] + uM * dphi[1] + uR * dphi[2])

            u_exact = exact_sol(x)
            h_eps = 1e-6
            du_exact = (exact_sol(x + h_eps) - exact_sol(x - h_eps)) / (2 * h_eps)

            err_val = u_exact - u_h
            err_grad = du_exact - u_h_prime

            error_sq += w * (err_val**2 + err_grad**2) * h

    return np.sqrt(error_sq / 6.0)


def plot_convergence(f, exact, Ms, method="uniform"):
    hs = []
    errors_H1 = []
    errors_L2 = []
    print("{:>10} {:>15}".format("h", f"Error in L2 and H1 norm"))
    print("=" * 30)
    for M in Ms:
        nodes = create_partition(0, 1, M, method)
        u, x_dofs = solve_poisson_fem(nodes, f)
        h = np.max(np.diff(nodes))
        err_L2 = compute_L2_error(u, x_dofs, exact)
        err_H1 = compute_H1_error(u, x_dofs, exact)
        hs.append(h)
        errors_H1.append(err_H1)
        errors_L2.append(err_L2)
    hs = np.array(hs)
    errors_H1 = np.array(errors_H1)
    errors_L2 = np.array(errors_L2)
    print(errors_H1)
    print(errors_L2)
    rates_H1 = np.log(errors_H1[1:] / errors_H1[:-1]) / np.log(hs[1:] / hs[:-1])
    rates_L2 = np.log(errors_L2[1:] / errors_L2[:-1]) / np.log(hs[1:] / hs[:-1])
    plt.figure(figsize=(8, 5))
    plt.loglog(hs, errors_L2, 'o-', label=f"L2 error",color = "red")
    plt.loglog(hs, errors_H1, 'o-', label=f"H1 error", color = "blue")
    plt.loglog(hs, hs**4, 'k--', label=f"O(h^4)", color = "red")
    plt.loglog(hs, hs**2, 'k--', label=f"O(h^2)",color = "blue")
    plt.xlabel("h (max element size)")
    plt.ylabel(f"L2 and H1 error")
    plt.grid(True, which="both", ls="--")
    plt.title(f"Convergence of FEM Solution (P2 elements) in H1 and L2 norm")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\nEstimated convergence rates:")
    for i in range(len(rates_H1)):
        print(f"M = {Ms[i]} --> {Ms[i+1]}: rate_H1 = {rates_H1[i]:.2f}, rate_L2 = {rates_L2[i]:.2f}")

def plot(f, exact, M, method="uniform"):
    nodes = create_partition(0, 1, M, method)
    u, x_dofs = solve_poisson_fem(nodes, f)
    u_exact = np.array([exact(x) for x in x_dofs])
    plt.figure(figsize=(8, 5))
    plt.plot(x_dofs, u_exact, 'b--', label='Exact solution')
    plt.plot(x_dofs, u, 'o-', color='orange', label='FEM (P2)')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"FEM vs Exact Solution (M={M}, {method} mesh)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    exact = lambda x: x**4 * (1 - x)**3
    f = lambda x: -(4*(3*x**2 * (1-x)**3 -3*x**3* (1-x)**2) - 3*(6*x**5 - 10*x**4 + 4*x**3))
    M = 9
    plot(f, exact, M)
    Ms = [4, 8, 16, 32, 64, 128]
    plot_convergence(f, exact, Ms)