from __future__ import annotations
import inspect
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
FEAT_DIM = 8  # d_in = d_out per CGMP theory
ETAS = np.geomspace(1, 1e-3, num=10).tolist()

DEPTHS = list(range(1, 40))
PLOT_RESULTS = True 

torch.manual_seed(11)  
np.random.seed(11)

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
            print(f"Skip {cls.__name__}{kws} #{idx}: {exc}")
            continue
        tag = f"{cls.__name__}[{kws.get('name', '') or '-'}] #{idx}"
        yield tag, data


# ╔═════════════════╗
# ║    Utilities    ║
# ╚═════════════════╝

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
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or DEVICE

    # Deep-copy `data` so we can re-wire the metric without mutating caller
    data = data.clone()
    num_nodes = data.num_nodes
    edge_index = data.edge_index.to(device)

    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)

    edge_weight = initial_edge_weight(data).to(device)

    kappa = lly_curvature_limit_free(
        edge_index, num_nodes, edge_weight, combinatorial_only=False
    )

    w_half = ricci_flow_half_step(edge_weight, kappa)

    edge_index, w_half = metric_surgery(edge_index, w_half)

    w_norm = row_normalise(edge_index, w_half, num_nodes)
    lap_vals = laplacian(edge_index, w_norm, num_nodes)  # = −w_norm
    minus_L = -lap_vals  # positive weights

    mean_kappa = incident_curvature(edge_index, kappa, num_nodes)
    rho = curvature_gate(mean_kappa)  # (N,)

    Phi_self = layer.phi_self.weight.detach()  # (d_out, d_in)
    Phi_neigh = layer.phi_neigh.weight.detach()  # (d_out, d_in)

    # Sparse Jacobian -> dense
    T_sparse = layer_jacobian_sparse(
        edge_index, minus_L, rho, Phi_self, Phi_neigh
    )
    return T_sparse.to_dense()  # (N·d_out, N·d_in)


# ╔════════════════════════════════════════╗
# ║    Oversquashing sweep on one graph    ║
# ╚════════════════════════════════════════╝

def sweep_graph(
    tag: str,
    data: Data,
    *,
    etas: Optional[List[float]] = None,
) -> List[tuple[str, int, float, float]]:
    """Perform the oversquashing sweep on a *single* graph instance."""

    etas = etas or ETAS

    layer = CurvatureGatedMessagePropagationLayer(
        in_channels=FEAT_DIM,
        out_channels=FEAT_DIM,
        bias=False,
        device=DEVICE,
    )
    layer.eval()  # inference‑only

    T0 = build_layer_operator(layer, data.to(DEVICE))  # (N·d, N·d)

    depth_ops: List[torch.Tensor] = []
    Tk = T0.clone()
    depth_ops.append(Tk)
    for _ in range(2, max(DEPTHS) + 1):
        Tk = Tk @ T0
        depth_ops.append(Tk)

    J_depth = torch.stack([depth_ops[d - 1].reshape(-1) for d in DEPTHS])

    records = []
    for eta in etas:
        for d_idx, d in enumerate(DEPTHS, start=1):
            idx_val = oversquashing_index(
                J_depth[:d_idx],
                eta=eta,
                assume_unbounded=False,
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
            print(f"Processing {tag} ... (diameter = {diam_map[tag]})")
            rows = sweep_graph(tag, g)
            all_rows.extend(rows)
        except Exception as ex:
            print(f"  {tag} failed: {ex}")

    if not all_rows:
        raise RuntimeError("No graphs processed successfully – giving up.")

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
            xscale="log",
            xlabel="Depth $D$",
            ylabel="Oversquashing Index $\\mathcal{S}_{\\eta}^{\\mathrm{OSQ}}$",
        )

        # dashed vertical line at diameter of G 
        for ax in g.axes.flatten():
            title_text = ax.get_title()
            if "=" in title_text:
                tag = title_text.split("=")[-1].strip()
                diam = diam_map.get(tag)
                if diam is not None:
                    ax.axvline(diam, linestyle="--", linewidth=1, color="grey")

        g.add_legend(
            title=r"$\eta$",
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            frameon=False,
        )

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
