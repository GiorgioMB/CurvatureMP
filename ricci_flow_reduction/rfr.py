import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from typing import Union

class RicciFlowReduction:
    def __init__(self, target_dimensionality:int, 
                 tolerance:float=1e-10, n_neighbors:int=10, 
                 alpha_ricci:float=0.5, ricci_iters:int = 50, 
                 grad_step:int = 1, verbose: Union[str, None] = None):
        """
        Parameters
        - target_dimensionality: int - The dimensionality of the reduced space.
        - tolerance: float - The tolerance for the Ricci flow algorithm.
        - n_neighbors: int - The number of neighbors to consider for the k-nearest neighbors graph.
        - alpha_ricci: float - The alpha parameter for the Ollivier-Ricci curvature computation.
        - ricci_iters: int - The number of iterations for the Ricci flow algorithm.
        - grad_step: int - The step size for the gradient descent in the Ricci flow algorithm.
        - verbose: Union[str, None] - The verbosity level for the Ricci flow algorithm. Accepts "INFO", "TRACE", "DEBUG", "ERROR".
        """
        self.target_dim = target_dimensionality
        self.tolerance = tolerance
        self.n_neighbors = n_neighbors
        self.alpha = alpha_ricci
        self.ricci_iters = ricci_iters
        self.grad_step = grad_step
        self.verbose = verbose
        self._assert_inits()
    
    def _assert_inits(self):
        assert isinstance(self.target_dim, int), "target_dimensionality must be an integer."
        assert self.target_dim > 0, "target_dimensionality must be greater than 0."
        assert isinstance(self.tolerance, float), "tolerance must be a float."
        assert self.tolerance > 0, "tolerance must be greater than 0."
        if isinstance(self.verbose, str):
            assert self.verbose in ["INFO", "TRACE", "DEBUG", "ERROR"], "verbose must be one of 'INFO', 'TRACE', 'DEBUG', 'ERROR'."
            if self.tolerance > 0.1 and self.verbose in ["INFO", "DEBUG"]:
                print("Warning: tolerance is high. Consider setting it to a lower value.")
        else:
            assert self.verbose is None, "verbose must be a string or None."
        assert isinstance(self.n_neighbors, int), "n_neighbors must be an integer."
        assert self.n_neighbors > 0, "n_neighbors must be greater than 0."
        assert isinstance(self.alpha, float), "alpha must be a float."
        assert 0 < self.alpha < 1, "alpha must be between 0 and 1."
        assert isinstance(self.ricci_iters, int), "ricci_iters must be an integer."
        assert self.ricci_iters > 0, "ricci_iters must be greater than 0."
        assert isinstance(self.grad_step, int), "grad_step must be an integer."
        assert self.grad_step > 0, "grad_step must be greater than 0."

    def _build_knn_graph(self, X):
        """
        Build a k-nearest neighbors graph from the data points.
        """
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X)
        N = X.shape[0]
        G = nx.Graph()
        for i in range(N):
            for j_idx, j in enumerate(indices[i]):
                if i != j:
                    G.add_edge(i, j, weight=distances[i, j_idx])
        return G

    def _compute_curvatures(self, G):
        """
        Compute Ollivier-Ricci curvature on the graph edges.
        """
        if isinstance(self.verbose, str):
            orc = OllivierRicci(G, alpha=self.alpha, verbose=self.verbose)
        else:
            orc = OllivierRicci(G, alpha=self.alpha)
        orc.compute_ricci_curvature()
        curvature_dict = nx.get_edge_attributes(orc.G, 'ricciCurvature')
        curvatures = np.array(list(curvature_dict.values()))
        return curvatures

    def _ricci_flow(self, G):
        """
        Perform Ricci flow on the graph.
        """
        if isinstance(self.verbose, str):
            orc = OllivierRicci(G, alpha=self.alpha, verbose=self.verbose)
        else:
            orc = OllivierRicci(G, alpha=self.alpha)
        orc.compute_ricci_flow(iterations=self.ricci_iters, step=self.grad_step, delta=self.tolerance)
        G_transformed = orc.G.copy()
        return G_transformed

    def _compute_new_embedding(self, G_transformed, store = False):
        """
        Compute new embedding from the transformed graph using Diffusion Maps.
        """
        # Compute the shortest path distances in the transformed graph
        N = G_transformed.number_of_nodes()
        lengths = dict(nx.all_pairs_dijkstra_path_length(G_transformed, weight='weight'))
        distance_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                distance_matrix[i, j] = lengths[i][j]

        # Compute the kernel matrix using the Gaussian kernel
        sigma = np.mean(distance_matrix)
        kernel_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))

        # Perform eigen decomposition
        D = np.sum(kernel_matrix, axis=1)
        D_inv = np.diag(1 / D)
        M = D_inv @ kernel_matrix @ D_inv  # Markov matrix

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(M)
        if store:
            self.eigvals_ = eigvals
            self.eigvecs_ = eigvecs
            self.sigma_ = sigma
            self.D_inv_ = D_inv

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(-eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Skip the first eigenvector (corresponding to the largest eigenvalue 1)
        X_reduced = eigvecs[:, 1:self.target_dim+1] * eigvals[1:self.target_dim+1]
        return X_reduced

    def fit_transform(self, X):
        G = self._build_knn_graph(X)
        curvatures = self._compute_curvatures(G)
        max_curvature = np.max(curvatures)
        min_curvature = np.min(curvatures)

        # Avoid division by zero
        if min_curvature == 0:
            curvature_ratio = np.inf
        else:
            curvature_ratio = max_curvature / min_curvature

        if curvature_ratio < 4:
            G_transformed = self._ricci_flow(G)
        else:
            if isinstance(self.verbose, str):
                print("Ricci flow condition not met. Proceeding without Ricci flow.")
            G_transformed = G

        X_reduced = self._compute_new_embedding(G_transformed)
        return X_reduced
    
    def fit(self, X):
        """
        Fit the RicciFlowReduction model to the data.
        """
        self.X_train_ = X  # Store training data
        self.G_ = self._build_knn_graph(X)
        self.curvatures_ = self._compute_curvatures(self.G_)
        max_curvature = np.max(self.curvatures_)
        min_curvature = np.min(self.curvatures_)

        # Avoid division by zero
        if min_curvature == 0:
            self.curvature_ratio_ = np.inf
        else:
            self.curvature_ratio_ = max_curvature / min_curvature

        if self.curvature_ratio_ < 4:
            self.G_transformed_ = self._ricci_flow(self.G_)
        else:
            if isinstance(self.verbose, str):
                print("Ricci flow condition not met. Proceeding without Ricci flow.")
            self.G_transformed_ = self.G_

        # Compute the embedding and store necessary parameters for transformation
        self.X_reduced_ = self._compute_new_embedding(self.G_transformed_, True)

        return self

    def transform(self, X_new):
        """
        Transform the new data using the fitted model.
        """
        # Compute distances between X_new and X_train_
        distances = pairwise_distances(X_new, self.X_train_)

        # Compute the kernel matrix between new data and training data
        K_new = np.exp(-distances**2 / (2 * self.sigma_**2))

        # Compute D_new
        D_new = np.sum(K_new, axis=1)
        D_new_inv = 1 / D_new

        # Normalize K_new
        K_new_normalized = (D_new_inv[:, np.newaxis] * K_new) * self.D_inv_[np.newaxis, :]

        # Project onto the eigenvectors
        eigvecs = self.eigvecs_[:, 1:self.target_dim+1]
        eigvals = self.eigvals_[1:self.target_dim+1]

        X_new_reduced = (K_new_normalized @ eigvecs) * eigvals[np.newaxis, :]

        return X_new_reduced 
