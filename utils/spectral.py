import numpy as np

def laplacian(A, laplacian_type, tol=1):
    """ 
    Helper function for GSP methods
    A is a dense matrix 
    """
    n = A.shape[0]
    deg = A.sum(axis=1) # will be a (n,) array
    if laplacian_type == 'combinatorial':
        D = np.diag(deg)
        L = D - A
        return 0.5 * (L + L.T)  # Ensure symmetry
    elif laplacian_type == "random_walk":
        # I - A D^-1
        deg_inv = np.zeros_like(deg, dtype=A.dtype)
        nonzero = deg > tol
        deg_inv[nonzero] = 1.0 / deg[nonzero]
        P = A * deg_inv[None, :] # column stochastic (right mat mul)
        return np.eye(n, dtype=A.dtype) - P
    
    elif laplacian_type == "symmetric_normalized":
        deg_inv_sqrt = np.zeros_like(deg, dtype=A.dtype)
        nonzero = deg > tol
        deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
        S = (deg_inv_sqrt[:, None] * A) * deg_inv_sqrt[None, :]  # D^{-1/2} A D^{-1/2}
        L = np.eye(n, dtype=A.dtype) - S

        # New: make the Laplacian disconnected where graph is disconnected
        isolated_nodes = ~nonzero
        L[isolated_nodes, :] = 0
        L[:, isolated_nodes] = 0

        return 0.5 * (L + L.T)  # Ensure symmetry
    raise ValueError(f'Unknown laplacian type: {laplacian_type}')



def find_non_trivial_eigenvectors(w, v, tol=1e-8):
    """
    Returns all eigenvectors up to two non-trivial eigenvectors, where a 
    trivial eigenvector is defined as an eigenvector corresponding to 
    a 0-eigenvalue (Laplacian)

    Note that the first 0-eigenvalue will never be returned because
    this corresponds to the trivial eigenvector of all ones. 

    The second 0-eigenvalue onwards is returned because it can be a perfect cut (e.g. disconnected graph)
    """
    non_trivial = np.where(np.abs(w) > tol)[0]

    if len(non_trivial) < 2:
        raise ValueError(f"Not enough non-trivial eigenvectors found. "
                        f"Only {len(non_trivial)} with eigenvalue > {tol}. "
                        f"Recommed setting compartment=False")

    eigvecs = []
    for idx in range(1, non_trivial[0]):
        eigvecs.append(v[:, idx])
    eigvecs.append(v[:, non_trivial[0]])
    eigvecs.append(v[:, non_trivial[1]])
    return eigvecs


def compute_derivative_eigvectors(eigvecs):
    eigdiff = []
    for vec in eigvecs:
        eigdiff.append(np.array([0] + list(np.abs(np.diff(vec)))))
    return eigdiff