# Ricci Flow Reduction
`ricci_flow_reduction` is a Python library for dimensionality reduction that uses the concept of Ricci flow to smooth and adjust the geometry of a graph constructed from high-dimensional data. The library is well-suited for tasks such as manifold learning, where preserving the global structure of the data is essential. By leveraging Ollivier-Ricci curvature and Ricci flow, it provides a novel way to reduce dimensionality while accounting for the intrinsic curvature of the data manifold.

## Overview

### Ricci Flow in Graphs

Ricci flow is a process that adjusts the geometric structure of a manifold (or in this case, a graph) by evolving its curvature over time. For a graph $G = (V, E)$, where $V$ represents the nodes and $E$ represents the edges, Ricci flow smooths out the edge weights based on the Ollivier-Ricci curvature $\kappa$.

The Ollivier-Ricci curvature $\kappa$ of an edge $(i, j)$ is defined as:

$$\kappa(i, j) = 1 - \frac{W_1(\mu_i, \mu_j)}{d(i, j)}$$

where $W_1$ is the Wasserstein-1 distance (earth mover's distance) between local distributions $\mu_i$ and $\mu_j$ centered at nodes $i$ and $j$, and $d(i, j)$ is the distance between nodes $i, j$.

### When is Surgery Applied?

During the Ricci flow process, edge weights $w_{ij}$ are adjusted iteratively according to the curvature. However, in cases where the curvature becomes extreme (positive or negative), simple Ricci flow may not converge or could lead to numerical instability. In such cases, **surgery** is performed to stabilize the process.

- **Surgery Condition**: Surgery is applied when the ratio of the maximum to minimum absolute Ricci curvature exceeds a predefined threshold, denoted as $\kappa_{ratio}$:

$$\kappa_{ratio} = \frac{\max |\kappa(i, j)|}{\min |\kappa(i, j)|} \geq 4$$
- **Surgery Mechanism**: During surgery, edge weights for edges with extreme curvatures are adjusted towards their initial values to prevent further divergence. Specifically, for an edge $(i, j) $ with curvature $\kappa(i, j)$ exceeding a threshold $\kappa_{threshold}$:

$$w_{ij}' = \lambda \cdot w_{ij} + (1 - \lambda) \cdot w_{ij}^0$$

where $w_{ij}^0$ is the initial edge weight, and $\lambda$ is a scaling factor:

$$\lambda = \min\left(\frac{\kappa_{threshold}}{|\kappa(i, j)|}, 1 \right)$$

This adjustment ensures that edges with excessive curvature are gradually brought back towards their initial structure, allowing the Ricci flow to proceed smoothly.

### Dimensionality Reduction Process

1. **k-NN Graph Construction**: A k-nearest neighbors (k-NN) graph is built from the input data $X \in \mathbb{R}^{n \times d}$, where each node represents a data point and edges represent distances between neighbors.
2. **Ricci Curvature Calculation**: Compute the Ollivier-Ricci curvature for each edge $(i, j)$.
3. **Ricci Flow Application**: Iteratively adjust the edge weights using Ricci flow, performing surgery when necessary.
4. **Embedding Calculation**: Use the transformed graph structure to compute a diffusion map, yielding the low-dimensional embedding $X_{reduced} \in \mathbb{R}^{n \times d_{target}}$.

## Installation

To install the library, run:

```bash
pip install ricci_flow_reduction
```

## Usage Example

Here's an example of using `RicciFlowReduction` on a Swiss roll dataset, which is a common test for non-linear dimensionality reduction methods.

### Example: Swiss Roll

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from ricci_flow_reduction import RicciFlowReduction

# Generate Swiss roll dataset
n_samples = 1000
X, color = make_swiss_roll(n_samples, noise=0.1)

# Instantiate the RicciFlowReduction model
ricci_flow = RicciFlowReduction(
    target_dimensionality=2, 
    n_neighbors=10, 
    alpha_ricci=0.5, 
    ricci_iters=50, 
    grad_step=1, 
    verbose='INFO'
)

# Fit and transform the data
X_reduced = ricci_flow.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title('Swiss Roll - Ricci Flow Reduction')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar()
plt.show()
```

In this example:
- A Swiss roll dataset is generated using `sklearn.datasets.make_swiss_roll`, which simulates a 3D non-linear manifold.
- The `RicciFlowReduction` model is applied to transform this data into a 2D space.
- The results are visualized using a scatter plot, where points are colored according to their position in the original Swiss roll, illustrating how the Ricci flow captures the intrinsic structure.

## Parameters

- `target_dimensionality` (int): Target dimensionality for the reduced space.
- `tolerance` (float, default=1e-10): Convergence tolerance for Ricci flow.
- `n_neighbors` (int, default=10): Number of neighbors for the k-NN graph.
- `alpha_ricci` (float, default=0.5): Alpha parameter for Ollivier-Ricci curvature computation. $ 0 < lpha < 1 $.
- `ricci_iters` (int, default=50): Number of iterations for Ricci flow.
- `grad_step` (int, default=1): Step size for gradient adjustment in Ricci flow.
- `verbose` (str or None, default=None): Verbosity level (`'INFO'`, `'TRACE'`, `'DEBUG'`, `'ERROR'`) or `None` for silent mode.

## Methods

### `fit_transform(X)`

Fits the RicciFlowReduction model to data $ X $ and returns the reduced dimensionality representation.

**Parameters:**
- `X` (np.ndarray): Input data of shape $(n_{samples}, n_{features})$.

**Returns:**
- `X_reduced` (np.ndarray): Transformed data of shape $(n_{samples}, target\_dimensionality)$.

### `fit(X)`

Fits the model to the input data $ X $ without returning the reduced representation directly.

### `transform(X_new)`

Transforms new data points $ X_{new} $ using the fitted model.

**Parameters:**
- `X_new` (np.ndarray): New data points of shape $(n_{samples}, n_{features})$.

**Returns:**
- `X_new_reduced` (np.ndarray): Transformed data in the reduced space.

## Theory Behind the Method

### Ricci Flow

Ricci flow is analogous to smoothing out the geometry of a surface. In the context of graphs, this involves iteratively adjusting edge weights to reflect changes in curvature. It aims to reduce distortions in the graph structure, helping to preserve the manifold's shape in a lower-dimensional space.

### Diffusion Maps

After transforming the graph with Ricci flow, the diffusion map method is applied. This method approximates the eigenvectors of a Markov matrix constructed from the adjusted graph, capturing the essential structure of the data and providing the low-dimensional embedding.

### Ricci Flow with Surgery

Surgery stabilizes the Ricci flow when curvature values deviate significantly. It prevents the algorithm from diverging by adjusting edge weights to ensure that extreme curvature differences do not overly distort the graph's geometry.

By combining these steps, the `RicciFlowReduction` class offers a powerful approach to extract meaningful low-dimensional representations from high-dimensional data while retaining the essential manifold structure.

## License

`ricci_flow_reduction` is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request on the [GitHub repository](https://github.com/your-repo/ricci_flow_reduction).
