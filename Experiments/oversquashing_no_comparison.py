from __future__ import annotations
import inspect
from math import isnan
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx  # for graph diameter
import numpy as np  # granular eta sweep
import pandas as pd
import seaborn as sns
import torch
import torch_geometric.datasets as pyg_datasets
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected
from Layers.utils import (
    curvature_gate,
    incident_curvature,
    layer_jacobian_sparse,
    laplacian,
    lly_curvature_limit_free,
    metric_surgery,
    oversquashing_index,
    ricci_flow_half_step,
    row_normalise,
)
from Layers.cgmp_layer import CurvatureGatedMessagePropagationLayer 
torch.manual_seed(123)  
np.random.seed(123)

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

# ╔════════════════════════════════╗
# ║ Hyper‑Parameters and Settings  ║
# ╚════════════════════════════════╝
ROOT = "data"  # download/cache root for PyG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ETAS = np.geomspace(1, 1e-4, num=15).tolist()

DEPTHS = list(range(1, 40))
PLOT_RESULTS = True 


# ╔════════════════════════╗
# ║       Benchmarks       ║       
# ╚════════════════════════╝
SELECTED_GRAPHS: List[Tuple[type, dict, int]] = [
    (pyg_datasets.TUDataset, {"name": "ENZYMES"}, 1),
    (pyg_datasets.TUDataset, {"name": "ENZYMES"}, 2),
    (pyg_datasets.KarateClub, {}, 0),
    (pyg_datasets.TUDataset, {"name": "MUTAG"}, 0),
    (pyg_datasets.TUDataset, {"name": "MUTAG"}, 1),
    (pyg_datasets.TUDataset, {"name": "MUTAG"}, 2),
]


# ╔══════════════════════════════════╗
# ║  Helper: Dataset Instantiation   ║       
# ╚══════════════════════════════════╝

def _instantiate_dataset(cls, root: str, kwargs: dict):
    if "root" in inspect.signature(cls.__init__).parameters:
        root_dir = Path(root) / cls.__name__.lower() / kwargs.get("name", "")
        return cls(root=str(root_dir), **kwargs)
    return cls(**kwargs)


def iter_selected_graphs(root: str = ROOT):
    for cls, kws, idx in SELECTED_GRAPHS:
        try:
            ds = _instantiate_dataset(cls, root, kws)
            data = ds[idx]
        except Exception as exc:
            print(f"✗ Skip {cls.__name__}{kws} #{idx}: {exc}")
            continue
        tag = f"{cls.__name__}[{kws.get('name', '') or '-'}] #{idx}"
        yield tag, data


# ╔═════════════════╗
# ║    Utilities    ║
# ╚═════════════════╝

def distance_matrix(data: Data) -> torch.Tensor:
    """Return an (N, N) tensor of unweighted shortest-path lengths."""
    edge_index = data.edge_index.cpu()
    N = data.num_nodes
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edge_index.t().numpy())
    # Floyd–Warshall is fine for graphs ≤ ~2000 nodes
    dist = torch.from_numpy(nx.floyd_warshall_numpy(G)).long()
    return dist

def graph_diameter(data: Data) -> int:
    """Return the diameter of the (undirected) graph underlying *data*.

    If the graph is disconnected, take the maximum diameter among its
    connected components.
    """

    edge_index = data.edge_index.cpu()
    num_nodes = data.num_nodes

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().numpy())

    if nx.is_connected(G):
        return nx.diameter(G)
    return max(nx.diameter(G.subgraph(c)) for c in nx.connected_components(G))



def _l1_normalise(w: torch.Tensor) -> torch.Tensor:
    w = w.abs()
    return w.float() / w.sum().clamp_min(1e-18)


def initial_edge_weight(data: Data) -> torch.Tensor:
    if getattr(data, "edge_weight", None) is not None:
        return _l1_normalise(data.edge_weight)

    if getattr(data, "edge_attr", None) is not None:
        attr = data.edge_attr.float()
        if attr.dim() == 1 or attr.size(-1) == 1:
            w = attr.view(-1)
        else:  # vector features
            w = attr.abs().sum(dim=-1)  # L1
        return _l1_normalise(w)

    # completely unweighted
    E = data.edge_index.size(1)
    return torch.full((E,), 1.0 / E, dtype=torch.float32)

# ╔════════════════════════════╗
# ║ Core: Build Dense Jacobian ║
# ╚════════════════════════════╝

def build_layer_operator(
    layer: CurvatureGatedMessagePropagationLayer,
    data: Data,
    w_norm: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.LongTensor] = None,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or DEVICE
    if torch.isnan(edge_index).any() or torch.isinf(edge_index).any():
        raise ValueError("NaN or Inf detected in edge_index.")
    if torch.isnan(w_norm).any() or torch.isinf(w_norm).any():
        raise ValueError("NaN or Inf detected in edge weights.")


    # Deep-copy `data` so we can re-wire the metric without mutating caller
    data = data.clone()
    num_nodes = data.num_nodes

    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)


    kappa = lly_curvature_limit_free(
        edge_index, num_nodes, w_norm#, combinatorial_only=True
    )

    lap_vals = laplacian(edge_index, w_norm, num_nodes)  # = −w_norm
    minus_L = -lap_vals  # positive weights

    mean_kappa = incident_curvature(edge_index, kappa, num_nodes)
    rho = curvature_gate(mean_kappa)  # (N,)

    Phi_self = layer.phi_self.weight.detach()  # (d_out, d_in)
    Phi_neigh = layer.phi_neigh.weight.detach()  # (d_out, d_in)

    ##check that everything fed to the jacobian is not NaN or Inf
    if torch.isnan(minus_L).any() or torch.isinf(minus_L).any():
        raise ValueError("NaN or Inf detected in Laplacian values.")
    if torch.isnan(rho).any() or torch.isinf(rho).any():
        print(f"Is NaN? {torch.isnan(rho).any()}")
        raise ValueError("NaN or Inf detected in curvature gate values.")
    if torch.isnan(Phi_self).any() or torch.isinf(Phi_self).any():
        raise ValueError("NaN or Inf detected in Phi_self weights.")
    if torch.isnan(Phi_neigh).any() or torch.isinf(Phi_neigh).any():
        raise ValueError("NaN or Inf detected in Phi_neigh weights.")
    if torch.isnan(edge_index).any() or torch.isinf(edge_index).any():
        raise ValueError("NaN or Inf detected in edge_index.")
    # Sparse Jacobian -> dense
    T_sparse = layer_jacobian_sparse(
        edge_index, minus_L, rho, Phi_self, Phi_neigh
    )
    return T_sparse.to_dense()  # (N * d_out, N * d_in)


# ╔════════════════════════════════════════╗
# ║    Oversquashing sweep on one graph    ║
# ╚════════════════════════════════════════╝
def _fake_loss(x: torch.Tensor) -> float:
    """Dummy loss function to use with autograd.jacobian."""
    return x.sum()  # just a placeholder, we don't care about the value


def sweep_graph(
    tag: str,
    data: Data,
    *,
    etas: Optional[List[float]] = None,
) -> List[tuple[str, int, float, float]]:
    etas = etas or ETAS
    FEAT_DIM = data.x.size(1) if data.x is not None else 1
    initial_x = data.x.clone().detach() if data.x is not None else None
    layer = CurvatureGatedMessagePropagationLayer(
        in_channels=FEAT_DIM,
        out_channels=FEAT_DIM,
        bias=False,
        device=DEVICE,
    )
    h, edge_index, weights = layer(data.x.to(DEVICE), data.edge_index.to(DEVICE), initial_x=initial_x)
    

    # One-step Jacobian
    T0 = build_layer_operator(layer, data.to(DEVICE), weights, edge_index)  # (N*d, N*d)

    # Pre-compute hop distances (CPU tensor is fine)
    dist = distance_matrix(data)

    # Power stack  T, T^2, ...   ->  full Jacobians for each depth
    depth_ops: List[torch.Tensor] = []
    depth_ops.append(T0.clone())  
    for _ in range(2, max(DEPTHS) + 1):
        print(f"Computing depth {_:2d} Jacobian...", end="\r")
        h, edge_index, weights = layer(h, edge_index, edge_weight=weights, initial_x=initial_x)
        Tk = build_layer_operator(layer, data.to(DEVICE), weights, edge_index)  # (N*d, N*d)
        if Tk.shape != T0.shape:
            raise ValueError(
                f"Shape mismatch: T0 {T0.shape} vs Tk {Tk.shape}. "
                "Ensure the layer is correctly set up."
            )
        ##check for naNs
        if torch.isnan(Tk).any():
            raise ValueError(
                f"NaN detected in Jacobian at depth {_}. "
                "Check the layer setup and input data."
            )
        ##check for infs
        if torch.isinf(Tk).any():
            raise ValueError(
                f"Inf detected in Jacobian at depth {_}. "
                "Check the layer setup and input data."
            )
        # accumulate powers of T
        if depth_ops:
            # Multiply the last operator by Tk
            Tk = torch.matmul(depth_ops[-1], Tk)
        depth_ops.append(Tk)

    J_depth = torch.stack(depth_ops)           # (maxL, N·d, N·d)

    records = []
    for eta in etas:
        for d_idx, d in enumerate(DEPTHS, start=1):
            print(f"Computing oversquashing index for {tag} at depth {d} with eta={eta:.2e} ...", end="\r")
            idx_val = oversquashing_index(
                J_depth[:d_idx], dist, eta=eta
            )
            records.append((tag, d, eta, idx_val))

    return records


# ╔══════════════════════════╗
# ║       Main driver        ║
# ╚══════════════════════════╝

def main():
    all_rows: List[tuple[str, int, float, float]] = []
    diam_map: dict[str, int] = {}

    for tag, g in iter_selected_graphs():
        try:
            diam_map[tag] = graph_diameter(g)
            print(f"Processing {tag} ... (diameter = {diam_map[tag]})            ")
            rows = sweep_graph(tag, g)
            all_rows.extend(rows)
        except Exception as ex:
            print(f"{tag} failed: {ex}")

    if not all_rows:
        raise RuntimeError("No graphs processed successfully -- giving up.")

    df = pd.DataFrame(all_rows, columns=["graph", "depth", "eta", "os_index"])
    csv_path = Path("oversquashing_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    if PLOT_RESULTS:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        g = sns.relplot(
            data=df,
            x="depth",
            y="os_index",
            hue="eta",
            col="graph",
            col_wrap=2,
            kind="line",
            linewidth=2,
            palette="colorblind",
            facet_kws=dict(sharex=True, sharey=False),
        )
        g.set(
            xlabel="Depth $D$",
            ylabel="Oversquashing Index $\\mathcal{S}_{\\eta}^{\\mathrm{OSQ}}$",
            yscale="log",
        )

        # dashed vertical line at diameter of G 
        for ax in g.axes.flatten():
            title_text = ax.get_title()
            if "=" in title_text:
                tag = title_text.split("=")[-1].strip()
                diam = diam_map.get(tag)
                if diam is not None:
                    ax.axvline(diam, linestyle="--", linewidth=1, color="grey")



        g.figure.subplots_adjust(top=0.9)
        g.figure.suptitle(
            "Oversquashing behaviour of a single CGMP layer\n(log-spaced $\\eta$; dashed line = diam$(G)$)"
        )

        fig_path = Path("oversquashing_sweep.png")
        g.savefig(fig_path, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
