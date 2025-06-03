import os, random, warnings, inspect
from typing import List, Dict, Tuple
import torch
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch_geometric.datasets as pyg_datasets
from torch_geometric.data import Data
from CGMP.layer import CurvatureGatedMessagePropagationLayer
from CGMP.utils import (
    layer_jacobian_sparse, depth_jacobian, oversquashing_index, _is_undirected,
    lly_curvature_limit_free, ricci_flow_half_step, cfl_delta_t, metric_surgery,
    row_normalise, laplacian, incident_curvature, curvature_gate,
)

plt.style.use("seaborn-v0_8-paper")
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.4,
    rc={
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
    }
)


DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT        = "data"
MAX_LAYERS  = 150
ETA         = 1e-5                      
PLOT_NAME   = "osq_finite_time_pure"

# ----- 1. graph helpers ------------------------------------------------------
GRAPHS = [
    (pyg_datasets.TUDataset, {"name": "ENZYMES"}, 1),
    (pyg_datasets.TUDataset, {"name": "ENZYMES"}, 2),
    (pyg_datasets.KarateClub,               {},  0),
    (pyg_datasets.TUDataset, {"name": "MUTAG"},   0),
    (pyg_datasets.TUDataset, {"name": "MUTAG"},   1),
    (pyg_datasets.TUDataset, {"name": "MUTAG"},   2),
]


def _instantiate_dataset(cls, root, kw):
    """Create a dataset instance, handling roots uniformly."""
    if "root" in inspect.signature(cls.__init__).parameters:
        root_dir = os.path.join(root, cls.__name__.lower(), kw.get("name", ""))
        return cls(root=root_dir, **kw)
    return cls(**kw)


def initial_x(data: Data):
    """Return an initial feature matrix.  Use existing `x` if present; otherwise
    fall back to one‑hot degree encoding."""
    if getattr(data, "x", None) is not None:
        return data.x.float()
    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float()
    return torch.nn.functional.one_hot(deg.long()).float()


def graph_diameter(data: Data) -> int:
    """Compute the (unweighted) diameter of the underlying graph."""
    r, c = data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(zip(r.tolist(), c.tolist()))
    return nx.diameter(G)


def iter_graphs():
    """Yield `(tag, Data)` pairs for the graphs in `GRAPHS`, restricted to the
    largest connected component when necessary."""
    for cls, kw, idx in GRAPHS:
        try:
            ds = _instantiate_dataset(cls, ROOT, kw)
            data = ds[idx]
        except Exception as e:
            warnings.warn(f"Skip {cls.__name__}{kw} #{idx}: {e}")
            continue

        # keep the giant component
        r, c = data.edge_index.cpu().numpy()
        G = nx.Graph(); G.add_nodes_from(range(data.num_nodes)); G.add_edges_from(zip(r, c))
        if not nx.is_connected(G):
            comp = max(nx.connected_components(G), key=len)
            mask = torch.tensor([v in comp for v in range(data.num_nodes)])
            rel = {old: i for i, old in enumerate(sorted(comp))}
            new_edges = [[rel[u], rel[v]] for u, v in zip(r, c) if u in comp and v in comp]
            data = Data(edge_index=torch.tensor(new_edges, dtype=torch.long).t().contiguous(),
                        x=data.x[mask] if data.x is not None else None)

        yield f"{cls.__name__}[{kw.get('name','')}] #{idx}", data.to(DEVICE)

def cgmp_layer_jacobian(layer,                       # CurvatureGatedMessagePropagationLayer
                        x: torch.Tensor,             # (N,d)
                        edge_index: torch.LongTensor,
                        edge_weight: torch.Tensor | None):
    n = x.size(0)
    no_w = edge_weight is None
    if no_w:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=x.device)
    is_undir = _is_undirected(edge_index, n)

    # Ricci flow half‑step
    kappa   = lly_curvature_limit_free(edge_index, n, edge_weight, combinatorial_only=(no_w and is_undir))
    dt      = cfl_delta_t(kappa, edge_weight)
    w_half  = ricci_flow_half_step(edge_weight, kappa, dt)
    edge_index, w_half = metric_surgery(edge_index, w_half)

    # Row‑normalised weights and Laplacian
    w_norm   = row_normalise(edge_index, w_half, n)
    minus_L  = -laplacian(edge_index, w_norm, n)
    mean_kap = incident_curvature(edge_index, kappa, n)
    rho      = curvature_gate(mean_kap)

    # Φ‑matrices
    Phi_s = layer.phi_self.weight       # (d_out, d_in)
    Phi_n = layer.phi_neigh.weight

    J_sparse = layer_jacobian_sparse(edge_index, minus_L, rho, Phi_s, Phi_n)
    return J_sparse.to_dense(), edge_index, w_half


# ----- 2. experiment loop ----------------------------------------------------

def run_one(seed: int) -> pd.DataFrame:
    random.seed(seed); torch.manual_seed(seed)
    recs: List[Dict] = []

    for gtag, data in iter_graphs():
        print(f"Running {gtag} seed={seed}…")
        D = graph_diameter(data)
        in_dim = initial_x(data).shape[1]
        x0 = initial_x(data).to(DEVICE)

        # CGMP stack – bare construction
        L_stack = min(MAX_LAYERS, 2 * D)
        cgmp_layers = torch.nn.ModuleList([
            CurvatureGatedMessagePropagationLayer(in_dim, in_dim).to(DEVICE)
            for _ in range(L_stack)
        ])

        Js_cgmp: List[float] = []          # spectral norm of depth‑L Jacobian
        Jlayers_c: List[torch.Tensor] = [] # list of layer Jacobians for prod

        # state holders
        h_c = x0
        eidx_c = data.edge_index
        w_c    = getattr(data, "edge_weight", torch.ones(eidx_c.size(1), device=DEVICE))

        for L in range(1, L_stack + 1):
            # forward pass through one CGMP layer to update hidden state
            layer = cgmp_layers[L - 1]
            if L > 1:  # skip propagation before layer‑0 Jacobian
                h_c, eidx_c, w_c = layer(h_c, eidx_c, w_c, initial_x=x0)

            # layer Jacobian
            Jc_L, eidx_c, w_c = cgmp_layer_jacobian(layer, h_c, eidx_c, w_c)
            Jlayers_c.append(Jc_L)
            depthJ_c = depth_jacobian(Jlayers_c)              # product up to depth L
            Js_cgmp.append(torch.linalg.norm(depthJ_c, ord=2).item())

        # oversquashing‑index series for CGMP
        S_c = [oversquashing_index(torch.tensor(Js_cgmp[:d], device=DEVICE).unsqueeze(1), eta=ETA)
                for d in range(1, len(Js_cgmp) + 1)]

        # Assertions: (i) monotone  (ii) saturation <= D
        assert all(S_c[i] >= S_c[i + 1] - 1e-9 for i in range(len(S_c) - 1)), \
            f"{gtag} seed={seed}: monotonicity fails"
        print(f"{gtag} seed={seed}: monotonicity OK")

        plateau = 1.0 / (1 + D)
        p_depth = next((i for i, s in enumerate(S_c, 1) if abs(s - plateau) < 1e-4), None)
        assert p_depth is not None and p_depth <= (D + 1), \
            f"{gtag} seed={seed}: no saturation by depth {D}"
        print(f"{gtag} seed={seed}: saturation by depth {p_depth} <= {D+1}")

        # log results
        for d, s in enumerate(S_c, 1):
            recs.append(dict(seed=seed, graph=gtag, depth=d, model="CGMP", S=s))

        print(f"{gtag} seed={seed} done.")
    return pd.DataFrame.from_records(recs)


# ----- 3. main ---------------------------------------------------------------

if __name__ == "__main__":
    df = pd.concat([run_one(sd) for sd in range(100)], ignore_index=True)
    df.to_csv("osq_pure_results.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.lineplot(
        data=df,
        x="depth",
        y="S",
        hue="graph",         
        estimator="mean",     
        errorbar="sd",        
        linewidth=2,
        ax=ax,
    )

    ax.set_xlabel("Layer depth $L$")
    ax.set_ylabel(fr"$\mathcal{{S}}_{{\eta}}\;(\eta={ETA})$")
    ax.set_title("Finite-Time Extinction of Oversquashing (CGMP)", weight="bold")
    ax.legend(title="Graph", frameon=False, loc="best")

    plt.tight_layout()
    fig.savefig(f"{PLOT_NAME}_combined.png", dpi=600, bbox_inches="tight")
    print("Combined plot saved.")
