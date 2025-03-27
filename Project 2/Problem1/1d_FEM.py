import numpy as np
import matplotlib.pyplot as plt

def create_partition(a, b, M, method="uniform"):
    """
    Create a partition of the interval [a, b] with M elements.
    method: "uniform", "squared", or "random"
    Returns: numpy array of M+1 node coordinates
    """
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
    """
    Quadratic (P2) shape functions on the reference element [0, 1].
    Returns shape function values [psi0, psi1, psi2] at xi.
    """
    psi0 = 2 * (xi - 0.5) * (xi - 1)
    psi1 = 4 * xi * (1 - xi)
    psi2 = 2 * xi * (xi - 0.5)
    return np.array([psi0, psi1, psi2])

def shape_function_derivatives_P2(xi):
    """
    Derivatives of P2 shape functions on the reference element [0, 1].
    Returns derivatives [dpsi0, dpsi1, dpsi2] at xi.
    """
    dpsi0 = 4 * xi - 3
    dpsi1 = 4 - 8 * xi
    dpsi2 = 4 * xi - 1
    return np.array([dpsi0, dpsi1, dpsi2])

def reference_to_physical(xi, a, b):
    """
    Affine mapping from reference element [0, 1] to physical element [a, b].
    xi: point in reference element
    a, b: endpoints of physical element
    Returns: x = Phi_K(xi)
    """
    return a + (b - a) * xi

def local_to_global_dof_index(k):
    """
    Local to global mapping for element k (P2 elements).
    Element k maps to global DOF indices: [2k, 2k+1, 2k+2]
    """
    return [2 * k, 2 * k + 1, 2 * k + 2]

def assemble_local_element_matrices(a, b, f):
    """
    Compute the local stiffness matrix A_K and load vector b_K
    on element [a, b] using Simpson's rule (3-point).
    """
    A_local = np.zeros((3, 3))
    b_local = np.zeros(3)
    
    # Quadrature points and weights for Simpson's rule
    quad_points = [0.0, 0.5, 1.0]
    quad_weights = [1.0, 4.0, 1.0]

    h = b - a                # Element length

    for xi, w in zip(quad_points, quad_weights):
        # Evaluate shape functions and derivatives at quadrature point
        phi = shape_functions_P2(xi)
        dphi = shape_function_derivatives_P2(xi)

        # Map quadrature point to physical element
        x = reference_to_physical(xi, a, b)
        f_val = f(x)

        # Compute contributions to stiffness matrix and load vector
        for i in range(3):
            for j in range(3):
                A_local[i, j] += w * dphi[i] * dphi[j] * (1/h)**2 * h
            b_local[i] += w * f_val * phi[i] * h

    # Apply Simpson's rule scaling
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

    # Apply Dirichlet BCs: u(0) = u(1) = 0
    # Remove first and last row/column of A and first/last entry of b
    A_reduced = A_global[1:-1, 1:-1]
    b_reduced = b_global[1:-1]
    return A_reduced, b_reduced, N_dof

def solve_poisson_fem(nodes, f):
    A_reduced, b_reduced, N_dof = assemble_global_system(nodes, f)
    u_reduced = np.linalg.solve(A_reduced, b_reduced)
    u_full = np.zeros(N_dof)
    u_full[1:-1] = u_reduced

    # Generate global FEM nodes: x0, x1, ..., x_{2M}
    x_dofs = [nodes[0]]
    for k in range(len(nodes) - 1):
        xL = nodes[k]
        xR = nodes[k + 1]
        mid = 0.5 * (xL + xR)
        x_dofs.extend([mid, xR])
    x_dofs = np.array(x_dofs)

    return u_full, x_dofs


def compute_L2_error(u_num, x_dofs, exact_sol):
    return np.sqrt(np.sum((u_num - exact_sol(x_dofs))**2) / len(x_dofs))

def plot_convergence(f, exact, Ms, method="uniform"):
    hs = []
    errors = []
    
    print("{:>10} {:>15}".format("h", "L2 Error"))
    print("=" * 26)
    
    for M in Ms:
        nodes = create_partition(0, 1, M, method)
        u, x_dofs = solve_poisson_fem(nodes, f)
        h = np.max(np.diff(nodes))  # max step size (non-uniform case support)
        err = compute_L2_error(u, x_dofs, exact)
        hs.append(h)
        errors.append(err)
        print(f"{h:10.4e} {err:15.8e}")

    hs = np.array(hs)
    errors = np.array(errors)
    rates = np.log(errors[1:] / errors[:-1]) / np.log(hs[1:] / hs[:-1])

    # Plot log-log convergence
    plt.figure(figsize=(8, 5))
    plt.loglog(hs, errors, 'o-', label="L2 error")
    plt.loglog(hs, hs**3, 'k--', label="O(h^3)")  # Expected for P2 elements
    plt.xlabel("h (max element size)")
    plt.ylabel("L2 error")
    plt.grid(True, which="both", ls="--")
    plt.title("Convergence of FEM Solution (P2 elements)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nEstimated convergence rates:")
    for i in range(len(rates)):
        print(f"M = {Ms[i]} --> {Ms[i+1]}: rate ≈ {rates[i]:.2f}")

    

if __name__ == "__main__":
    exact = lambda x: x**3 * (1 - x)**3
    f = lambda x: -6 * x + 36 * x**2 - 60 * x**3 + 30 * x**4



    Ms = [4, 8, 16, 32, 64, 128]
    plot_convergence(f, exact, Ms)
