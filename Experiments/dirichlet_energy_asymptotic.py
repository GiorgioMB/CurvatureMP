import os
import inspect
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, GCNConv
from typing import Dict, List, Iterator, Tuple
from torch_geometric.data import Dataset, InMemoryDataset
import torch_geometric.datasets as pyg_datasets
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.data import HeteroData
from ..CGMP.cgmp_layer import CurvatureGatedMessagePropagationLayer
from ..CGMP.utils import (
    row_normalise,
    dirichlet_energy,
)

import random
torch.manual_seed(0)
random.seed(0)

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
MAX_ITERS   = 5000          
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT        = "data"           # where datasets get downloaded

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


# ╔═════════════════════════========═══════════════════╗
# ║ 3. Training Loop for Dirichlet Energy Trajectories ║
# ╚══════════════════════════======════════════════════╝
def _initial_x(data: Data):
    if getattr(data, "x", None) is not None:
        return data.x.float()
    # fallback: one-hot degree
    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float()
    return torch.nn.functional.one_hot(deg.long()).float()

def run_dirichlet(model_tag, model, data: Data):
    """Return a list[E_dir] of length L_MAX for one model on one graph."""
    x = _initial_x(data).to(DEVICE)

    # architecture-specific book-keeping
    if model_tag == "CGMP":
        edge_index, edge_weight = data.edge_index, initial_edge_weight(data)
    else:                                  # GCN / GAT never mutate edges
        edge_index = data.edge_index
        edge_weight = row_normalise(edge_index,
                                   initial_edge_weight(data),
                                   data.num_nodes)

    traj = []
    h = x
    for layer in range(MAX_ITERS):
        if model_tag == "CGMP":
            h, edge_index, edge_weight = model(
                h, edge_index, edge_weight, initial_x=x)
        else:
            if model_tag == "GCN":
                h = model(h, edge_index, edge_weight)
            else:                           # GAT
                h = model(h, edge_index)    # attention uses edge_index only

        traj.append(
            dirichlet_energy(h, edge_index, edge_weight).item())
        if (layer+1) % 100 == 0:
            print(f"Layer {layer+1:04d} | Model {model_tag} | "
              f"Dirichlet Energy = {traj[-1]:.3e}")
    return traj


# ╔════════════════════════=═════════════════╗
# ║ 4. Full Experiment Run and Seaborn Plot  ║
# ╚══════════════════════════════════════════╝
MODELS = {
    "CGMP": lambda in_dim: CurvatureGatedMessagePropagationLayer(
                in_dim, in_dim).to(DEVICE),
    "GCN":  lambda in_dim: GCNConv(in_dim, in_dim, add_self_loops=True).to(DEVICE),
    "GAT":  lambda in_dim: GATConv(in_dim, in_dim, heads=2, concat=False).to(DEVICE),
}
def main():
    records = []   # rows for final DataFrame

    for graph_tag, g in iter_selected_graphs():
        in_dim = _initial_x(g).shape[1]
        for model_tag, ctor in MODELS.items():
            model = ctor(in_dim)
            try:
                traj = run_dirichlet(model_tag, model, g)
                records.extend(
                    dict(graph=graph_tag,
                         model=model_tag,
                         depth=i+1,
                         dirichlet=E)
                    for i, E in enumerate(traj)
                )
                print(f"{graph_tag} -- {model_tag}: done")
            except Exception as e:
                print(f"{graph_tag} -- {model_tag} failed: {e}")

    if not records:
        raise RuntimeError("Nothing processed!")

    df = pd.DataFrame.from_records(records)
    df.to_csv("dirichlet_energy_trajectories.csv", index=False)

    palette = sns.color_palette("colorblind", n_colors=df['model'].nunique())
    g = sns.relplot(
        data=df,
        x="depth",
        y="dirichlet",
        hue="model",
        style="graph",
        kind="line",
        linewidth=2,
        palette=palette,
        height=4,
        aspect=1.6,
    )
    g.set(xlabel="Layer depth L", ylabel="Dirichlet Energy", yscale="log")
    g.figure.suptitle("Dirichlet-Energy Decay: CGMP vs GCN vs GAT", weight="bold")
    g.savefig("dirichlet_energy_comparison.pdf", bbox_inches="tight")
    g.savefig("dirichlet_energy_comparison.png", dpi=600, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
