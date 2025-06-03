import os
import inspect
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import pkgutil
import itertools
from typing import Dict, List, Iterator, Tuple
from torch_geometric.data import Dataset, InMemoryDataset
import torch_geometric.datasets as pyg_datasets
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.data import HeteroData
from ..CGMP.utils import (
    lly_curvature_limit_free,
    ricci_flow_half_step,
    metric_surgery,
    curvature_variance_energy,
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

# -------------------------- hyper-parameters ---------------------------------
MAX_ITERS   = 100_000          # hard cap
CV_TOL      = 1e-12            # stop when curvature variance < tol
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT        = "data"           # where datasets get downloaded
#------------------------------------------------------------------------------

# ╔═══════════════════════===═══════════════════════╗
# ║ 1. Enumerate Selected Graphs in torch_geometric ║
# ╚══════════════════===════════════════════════════╝
SELECTED_GRAPHS = [
    (pyg_datasets.TUDataset,               {"name": "ENZYMES"},   1),
    (pyg_datasets.TUDataset,               {"name": "ENZYMES"},   2),
    (pyg_datasets.KarateClub,              {},                    0),  
    (pyg_datasets.TUDataset,               {"name": "MUTAG"},     0),
    (pyg_datasets.TUDataset,               {"name": "MUTAG"},     1),
    (pyg_datasets.TUDataset,               {"name": "MUTAG"},     2),
]

def _instantiate_dataset(cls, root: str, kwargs: dict):
    """
    Create a dataset instance, passing `root=` only if the class'
    __init__ actually accepts it (many toy sets like KarateClub don't).
    """
    if "root" in inspect.signature(cls.__init__).parameters:
        root_dir = os.path.join(root, cls.__name__.lower(),
                                kwargs.get("name", ""))
        return cls(root=root_dir, **kwargs)
    else:                                  
        return cls(**kwargs)


def iter_selected_graphs(
    root: str = "data",
    device: torch.device | None = None,
):
    """Yield (tag, data) for the 10 pre-selected graphs above."""
    if device is None:
        device = DEVICE

    for cls, kwargs, idx in SELECTED_GRAPHS:
        # download / cache on demand
        try:
            ds = _instantiate_dataset(cls, root, kwargs)
        except Exception as e:
            print(f"Skip {cls.__name__}{kwargs}: {e}")
            continue
        try:
            g = ds[idx]
        except IndexError:
            print(f"Skip {cls.__name__}{kwargs} #{idx}: index out of range")
            continue

        tag = f"{cls.__name__}[{kwargs.get('name', '')}] #{idx}"
        if isinstance(g, HeteroData):
            g = _to_homogeneous_safe(g)
            if g is None:
                continue
        yield tag, g.to(device)


# ╔══════════════════════════════════════════════════════╗
# ║ 2. Edge-Weight Assignment and Normalisation Helpers  ║
# ╚═══════════════════════════════════════════════=══════╝
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
        else:                           # vector features
            w = attr.abs().sum(dim=-1)  # L1
        return _l1_normalise(w)

    # completely unweighted
    E = data.edge_index.size(1)
    return torch.full((E,), 1.0 / E, dtype=torch.float32)


# ╔═════════════════════════==═══════════════════╗
# ║ 3. Ricci-Flow and Surgery on a Single Graph  ║
# ╚══════════════════════════════════════════════╝
def run_flow(tag: str, data: Data):
    edge_index = data.edge_index
    undirected_flag = is_undirected(edge_index)
    if not undirected_flag:
        edge_index, _ = to_undirected(edge_index)
    has_explicit_w     = getattr(data, "edge_weight", None) is not None
    has_edge_features  = getattr(data, "edge_attr",  None) is not None
    combinatorial_only = not has_explicit_w and undirected_flag
    print(f"Running Ricci flow on {tag} with combinatorial_only={combinatorial_only}")
    edge_weight = initial_edge_weight(data)
    edge_index  = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

    cv_history = []
    for _ in range(MAX_ITERS):
        kappa = lly_curvature_limit_free(edge_index,
                                      data.num_nodes,
                                      edge_weight, combinatorial_only=combinatorial_only)
        mean_kappa = kappa.mean().item()
        min_kappa  = kappa.min().item()
        max_kappa  = kappa.max().item()
        cv = curvature_variance_energy(kappa, edge_weight).item()
        if _ % 100 == 0 or cv < CV_TOL:
            print(
                    f"Iter {_+1:05d} | CV = {cv:.3e} | κ mean = {mean_kappa:.3e} | κ min = {min_kappa:.3e} | κ max = {max_kappa:.3e}"
                )
        cv_history.append(cv)

        if cv < CV_TOL:
            break

        edge_weight = ricci_flow_half_step(edge_weight, kappa)
        edge_index, edge_weight = metric_surgery(edge_index, edge_weight)
        edge_weight = _l1_normalise(edge_weight)

    print(f"{tag:>30}: {len(cv_history):>6} iters,  final CV = {cv_history[-1]:.2e}")
    return cv_history


# ╔════════════════════════=═════════════════╗
# ║ 4. Full Experiment Run and Seaborn Plot  ║
# ╚══════════════════════════════════════════╝
def main():
    trajectories = {}   # tag -> list[float]

    for tag, g in iter_selected_graphs():
        try:
            trajectories[tag] = run_flow(tag, g)
        except Exception as e:
            print(f"{tag} failed: {e}")

    if not trajectories:
        raise RuntimeError("No graphs processed successfully!")

    # ---- last-value-pad so they’re equally long --------------------
    L = max(len(v) for v in trajectories.values())
    padded = {k: v + [v[-1]] * (L - len(v)) for k, v in trajectories.items()}

    # ---- tidy -> dataframe ----------------------------------------
    df = pd.DataFrame(padded)
    df["iteration"] = range(L)
    df = df.melt(id_vars="iteration",
                 var_name="graph",
                 value_name="curv_var")
    df.to_csv("ricci_flow_curvature_variance.csv", index=False)
    graphs   = sorted(df["graph"].unique())
    palette  = sns.color_palette("colorblind", n_colors=len(graphs))

    # ---- seaborn viz ---------------------------------------------
    g = sns.relplot(
        data=df,
        x="iteration",
        y="curv_var",
        hue="graph",
        hue_order=graphs,
        palette=palette,
        kind="line",
        linewidth=2,
        height=4,          # inches
        aspect=1.6,        # width / height
        facet_kws=dict(sharex=True, sharey=False),
    )
    g.set(xlabel="Iteration", ylabel="Curvature Variance", yscale="log")
    g.figure.subplots_adjust(top=0.88)                 # space for title
    g.figure.suptitle("Ricci-Flow Curvature Variance", weight="bold")
    leg = g._legend
    leg.set_title(None)                                # no legend header
    leg.get_frame().set_linewidth(0)                   # frameless legend
    g.savefig("ricci_flow_curvature_variance.pdf", bbox_inches="tight")
    g.savefig("ricci_flow_curvature_variance.png", dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
