from __future__ import annotations
import inspect
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx  # for graph diameter
from networkx import is_connected
import numpy as np  # granular eta sweep
import pandas as pd
import seaborn as sns
import torch
import torch_geometric.datasets as pyg_datasets
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected, degree
from ..CGMP.utils import (
    curvature_gate,
    incident_curvature,
    layer_jacobian_sparse,
    oversquashing_index,
    depth_jacobian,
    lly_curvature_limit_free,
)
from ..CGMP.layer import CurvatureGatedMessagePropagationLayer 
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
HIGH_DECADES = [2, 4, 8]  ## this is to check for eta star * 10^HIGH_DECADES
DEPTHS = list(range(1, 1000))
PLOT_RESULTS = True 


# ╔════════════════════════╗
# ║       Benchmarks       ║       
# ╚════════════════════════╝
SELECTED_GRAPHS: List[Tuple[type, dict, int]] = [
    (pyg_datasets.KarateClub, {}, 0),
    (pyg_datasets.TUDataset, {"name": "ENZYMES"}, 1),
    (pyg_datasets.TUDataset, {"name": "ENZYMES"}, 2),
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
def make_eta_grid(eta_star: float,
                  span_decades: int = 6,
                  n_levels: int = 15
                  ) -> list[float]:
    if eta_star <= 0.0 or not np.isfinite(eta_star):
        raise ValueError("eta_star must be a positive, finite number.")
    high_decades = HIGH_DECADES
    # ---- 1. Downward log-space grid -----------------------------
    log10_eta_star = np.log10(eta_star)
    exps_down = np.linspace(0.0, -span_decades, n_levels)   
    log10_etas_down = log10_eta_star + exps_down

    # ---- 2. Upward points ---------------------------------
    exps_up = np.array(high_decades, dtype=float)          
    log10_etas_up = log10_eta_star + exps_up

    # ---- 3. Merge and exponentiate once, with float-safe clipping ----------
    log10_etas = np.concatenate([log10_etas_down, log10_etas_up])
    tiny64 = np.finfo(np.float64).tiny                       # 2.225e-308
    etas = np.power(10.0, log10_etas)
    etas = np.clip(etas, tiny64, None)

    # ---- 4. Deduplicate & sort ascending ---------------------------------
    etas_unique_sorted = np.unique(etas)
    return etas_unique_sorted.tolist()

def check_connectivity(data: Data) -> bool:
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().cpu().numpy())
    return nx.is_connected(G)

def distance_matrix(data: Data) -> torch.Tensor:
    edge_index = data.edge_index.cpu()
    N = data.num_nodes
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edge_index.t().numpy())
    # Floyd–Warshall call
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
    row_norm_weight: torch.Tensor,
    edge_index: torch.LongTensor,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or layer.phi_self.weight.device
    num_nodes = data.num_nodes

    # ------------------------------------------------------------------
    # Ensure the graph is undirected –  Jacobian formula requires it
    # ------------------------------------------------------------------
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)

    if torch.isnan(row_norm_weight).any() or torch.isinf(row_norm_weight).any():
        raise ValueError(
            "Row-normalised edge weights contain NaN or Inf values. "
            "This is likely due to a numerical instability in the "
            "curvature computation or Ricci flow step."
        )
    # ------------------------------------------------------------------
    #   Curvature gate 
    # ------------------------------------------------------------------
    kappa = lly_curvature_limit_free(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weight=None,
        combinatorial_only=True,
    )
    mean_kappa = incident_curvature(edge_index, kappa, num_nodes)
    rho = curvature_gate(mean_kappa) 

    # ------------------------------------------------------------------
    #   Laplacian entries for neighbour hops 
    # ------------------------------------------------------------------
    laplacian_vals = -row_norm_weight  # negative values

    Phi_self = layer.phi_self.weight.detach()
    Phi_neigh = layer.phi_neigh.weight.detach()

    T_sparse = layer_jacobian_sparse(
        edge_index=edge_index,
        laplacian_vals=laplacian_vals,
        rho=rho,
        Phi_self=Phi_self,
        Phi_neigh=Phi_neigh,
    )
    return T_sparse.to_dense()


# ╔════════════════════════════════════════╗
# ║    Oversquashing sweep on one graph    ║
# ╚════════════════════════════════════════╝
def sweep_graph(
    tag: str,
    data: Data,
    max_depth: int,
    device: torch.device,
    combinatorial_only: bool,
):
    # ------------------------------------------------------------------
    #   Node features -- must be non‑constant
    # ------------------------------------------------------------------
    data = data.to(device)
    if data.x is None:
        data.x = torch.randn(data.num_nodes, 8, device=device)
    initial_x = data.x.detach()
    feat_dim = data.x.size(1)

    # ------------------------------------------------------------------
    #   Rolling state for forward propagation
    # ------------------------------------------------------------------
    h = data.x
    edge_index = data.edge_index.to(device)
    weights = data.edge_weight

    # ------------------------------------------------------------------
    #   Lists that grow with depth
    # ------------------------------------------------------------------
    layer_ops: List[torch.Tensor] = []   # T_1, T_2, ...
    full_Jacobians: List[torch.Tensor] = []
    for depth in range(1, max_depth + 1):
        # ---- fresh CGMP layer -----------------------------------------
        layer_k = CurvatureGatedMessagePropagationLayer(
            in_channels=feat_dim,
            out_channels=feat_dim,
            bias=False,
            device=device,
           
        )
        h, edge_index, weights = layer_k(
            h,
            edge_index,
            edge_weight=weights,
             combinatorial_only = combinatorial_only,
            initial_x=initial_x,
        )
        ##check if edge_index is not empty
        if edge_index.numel() == 0:
            raise ValueError(
                f"Layer {tag} depth {depth} produced an empty edge_index."
            )
        ##check if weights are valid
        if weights is None or not torch.isfinite(weights).all():
            raise ValueError(
                f"Layer {tag} depth {depth} produced invalid edge weights."
            )
        ##check that everything is not NaN or Inf
        if torch.isnan(h).any() or torch.isinf(h).any():
            raise ValueError(
                f"Layer {tag} depth {depth} produced NaN or Inf values in node features."
            )
        if depth == 1:
            with torch.no_grad():
                w_min = weights.min().item()
                row, _ = edge_index  
                node_deg = degree(row, data.num_nodes, dtype=torch.long)
                delta_max = int(node_deg.max().item())
                tau = float(layer_k.tau.item())
                try:                                   
                    sigma_min = torch.linalg.svdvals(layer_k.phi_neigh.weight).min()
                except AttributeError:                  
                    sigma_min = torch.svd(layer_k.phi_neigh.weight, some=False)[1].min()
                kappa_n = float(sigma_min.item())
                D = graph_diameter(data)
                eta_star = tau * ((w_min/delta_max)**(D-1)) * (kappa_n**(D-1))
                print(f"Graph {tag} (D={D}): "
                      f"eta* = {eta_star:.3e} (tau={tau:.3f}, "
                      f"w_min={w_min:.3e}, delta_max={delta_max}, "
                      f"kappa_n={kappa_n:.3e})")
        Tk = build_layer_operator(
            layer=layer_k,
            data=data,
            row_norm_weight=weights,
            edge_index=edge_index,
            device=device,
        )
        layer_ops.append(Tk)

        # ---- full depth‑k Jacobian with teleport term ------------------
        Jk_full = depth_jacobian(layer_ops, tau=float(layer_k.tau.item()))
        full_Jacobians.append(Jk_full)

    jac_stack = torch.stack(full_Jacobians)  # (L, N*d_out, N*d_in)

    # ------------------------------------------------------------------
    #   Distance matrix
    # ------------------------------------------------------------------
    dist = distance_matrix(data.cpu())

    # ------------------------------------------------------------------
    #   Oversquashing index table
    # ------------------------------------------------------------------
    etas = make_eta_grid(eta_star, span_decades=6, n_levels=15)

    records = []
    for eta in etas:
        for depth_idx, depth in enumerate(range(1, max_depth + 1)):
            idx_val = oversquashing_index(jac_stack[: depth_idx + 1], dist, eta=eta)
            records.append((tag, depth, eta, idx_val))

    return records



# ╔══════════════════════════╗
# ║       Main driver        ║
# ╚══════════════════════════╝
def main():
    all_rows: List[tuple[str, int, float, float]] = []
    diam_map: dict[str, int] = {}

    for tag, g in iter_selected_graphs():
        try:
            ##fail for not karate club
            is_connected = check_connectivity(g)
            print(f"Processing {tag} ... (connected = {is_connected})")
            diam_map[tag] = graph_diameter(g)
            print(f"Processing {tag} ... (diameter = {diam_map[tag]})            ")
            ##check if graph has weights or not and if it's undirected
            undirected_flag = is_undirected(g.edge_index)
            if not undirected_flag:
                g.edge_index, _ = to_undirected(g.edge_index)
            has_explicit_w     = getattr(g, "edge_weight", None) is not None
            combinatorial_only = not has_explicit_w and undirected_flag
            edge_weight = initial_edge_weight(g)
            g.edge_weight = edge_weight
            rows = sweep_graph(tag, g, max_depth=max(DEPTHS), device=DEVICE, combinatorial_only=combinatorial_only)
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
        unique_etas = np.sort(df["eta"].unique())             # preserve logical ordering
        n_levels     = len(unique_etas)
        base_pal = sns.color_palette("colorblind") if n_levels <= 10 \
                else sns.color_palette("husl", n_levels)

        eta_pal = dict(zip(unique_etas, base_pal))

        # --- 2. Plot --------------------------------------------------------
        g = sns.relplot(
            data=df,
            x="depth",
            y="os_index",
            hue="eta",
            hue_order=unique_etas,          # guarantees stable colour ordering
            palette=eta_pal,                
            col="graph",
            col_wrap=3,
            kind="line",
            linewidth=2,
            aspect=1.4,
            facet_kws=dict(sharex=True, sharey=False),
        )
        g.set(
            xlabel="Depth $D$",
            ylabel="Oversquashing Index $\\mathcal{S}_{\\eta}^{\\mathrm{OSQ}}$",
            yscale="log",
        )
        ##hide the whole legend
        g._legend.remove()

        # --- 3. Diameter reference lines -----------------------------------
        for ax in g.axes.flatten():
            title_text = ax.get_title()
            if "=" in title_text:
                tag = title_text.split("=")[-1].strip()
                diam = diam_map.get(tag)
                if diam is not None:
                    ax.axvline(diam, linestyle="--", linewidth=1, color="grey")

        g.figure.subplots_adjust(top=0.9)
        g.figure.suptitle(
            "Oversquashing behaviour of a single CGMP layer\n"
            "(log-spaced $\\eta$; dashed line = diam$(G)$)"
        )

        fig_path = Path("oversquashing_sweep.png")
        pdf_path = Path("oversquashing_sweep.pdf")
        g.savefig(pdf_path, bbox_inches="tight")
        g.savefig(fig_path, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved {fig_path}, {pdf_path}")


if __name__ == "__main__":
    main()
