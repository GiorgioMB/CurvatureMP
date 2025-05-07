from typing import Optional
from regex import W
from torch import nn
import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import degree
from utils import compute_LLY_curvature, compute_ORF_step
class CurvatureGatedMP(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        dt: Optional[float] = None,
        flow: Optional[bool] = True,
        debug: Optional[bool] = False
    ):
        super().__init__(aggr='add')
        self.lin_self  = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.dt        = dt
        self.flow      = flow
        self.debug     = debug

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.LongTensor, 
                edge_weight: Optional[torch.Tensor] = None, 
                curvature: Optional[torch.Tensor] = None):
        num_nodes = x.size(0)
        row, col = edge_index


        # 1) compute curvature κ^{(k)} if not provided (this only happens in the first step)
        if curvature is None:
            curv_old = compute_LLY_curvature(
                edge_index, num_nodes,
                edge_weight=edge_weight,
            )
            curvature = curv_old


        # 2) half-step Ricci flow
        if self.flow and self.training:
            edge_index, w_half = compute_ORF_step(
                edge_index = edge_index,
                curvature = curvature,
                edge_weight = edge_weight,
                delta_t = self.dt
            )
            if self.debug:
                print(f"Removed edges: {col.size(0) - edge_index.size(1)} ")
            row, col = edge_index
            row_sum = scatter_add(w_half, row, dim=0, dim_size=num_nodes)
            w_half = torch.clamp(w_half, min=1e-10)
            w_norm  = w_half / (row_sum[row] + 1e-16)
            w_norm = torch.clamp(w_norm, min=1e-10)

            curv_new = compute_LLY_curvature(
                edge_index, num_nodes,
                edge_weight=w_half,
            )
            
        else:
            w_norm = torch.ones_like(row, dtype=torch.float32) if edge_weight is None else edge_weight.float()
            curv_new = curvature
            w_half = torch.ones_like(row, dtype=torch.float32) if edge_weight is None else edge_weight.float()

        # 5) calculate residual gate
        mean_curv = scatter_mean(curv_new, col, dim=0, dim_size=num_nodes)
        rho       = torch.sigmoid(mean_curv).unsqueeze(1)   


        # 6) build Laplacian message
        neigh_sum = self.propagate(
            edge_index,
            x=x,
            edge_weight=w_norm
        )
        deg_w     = scatter_add(w_norm, row, dim=0, dim_size=num_nodes)
        lap_self  = deg_w.unsqueeze(1) * self.lin_neigh(x) - neigh_sum


        # 7) curvature-gated combination + activation
        self_part  = (1 - rho) * self.lin_self(x)
        neigh_part = - lap_self
        out =  self_part + neigh_part
        return out, curv_new, w_half, edge_index

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1,1) * self.lin_neigh(x_j)

    def update(self, aggr_out):
        return aggr_out
