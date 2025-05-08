from typing import Optional
from torch import nn
import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean
class ResidualGNNLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__(aggr='add')
        self.lin_self  = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        neigh_sum = self.propagate(
            edge_index,
            x=x)
        self_part  = self.lin_self(x)
        out =  self_part + neigh_sum
        return out

    def message(self, x_j):
        return self.lin_neigh(x_j)

    def update(self, aggr_out):
        return aggr_out
