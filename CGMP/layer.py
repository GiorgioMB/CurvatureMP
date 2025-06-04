import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple
from Layers.utils import (
    lly_curvature_limit_free,
    cfl_delta_t,
    ricci_flow_half_step,
    metric_surgery,
    row_normalise,
    laplacian,
    incident_curvature,
    curvature_gate,
    allocate_tau_budget,
    _is_undirected,
)

try:
    from torch_scatter import scatter_add
except ImportError:
    scatter_add = None  

class CurvatureGatedMessagePropagationLayer(nn.Module):
    """One CGMP layer.

    Parameters
    ----------
    in_channels, out_channels : int
        Feature dimensions before and *after* the layer.  The theory
        assumes they coincide; if they differ we insert an optional
        linear adapter so that the residual teleport term is well-typed.

    r, delta : float, optional
        Self/neighbor capacity ratio r and contraction margin delta for
        the tau-budget allocator (Algorithm 1).  Defaults follow
        the paper: r = 2, delta = 0.10.
    tau : float or None, optional
        Fixed teleport weight.  If not passed, we derive (s_self, s_neigh, τ)
        on the fly from the first forward pass using the realised
        rho_max.
    spectral_caps : Optional[Tuple[float, float]]
        Optional explicit caps (s_self, s_neigh).  If given, we perform
        hard spectral normalisation of phi_self / phi_neigh after every
        optimiser step.  Otherwise norms are left unconstrained.
    bias : bool
        Whether to include bias terms in phi_self / phi_neigh.
    device : torch.device or str or None
        Device placement for parameters created inside the layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        r: float = 2.0,
        delta: float = 0.10,
        tau: Optional[float] = None,
        spectral_caps: Optional[Tuple[float, float]] = None,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = r
        self.delta = delta
        self._tau_fixed = tau is not None
        self.register_buffer("tau", torch.tensor(float(tau) if tau is not None else 0.0))

        self.phi_self = nn.Linear(in_channels, out_channels, bias=bias, device=device)
        self.phi_neigh = nn.Linear(in_channels, out_channels, bias=bias, device=device)

        self._spectral_caps = spectral_caps
        if spectral_caps is not None:
            s_self, s_neigh = spectral_caps
            self.phi_self = spectral_norm(self.phi_self)
            self.phi_neigh = spectral_norm(self.phi_neigh)
            self.register_buffer("_cap_self", torch.tensor(s_self))
            self.register_buffer("_cap_neigh", torch.tensor(s_neigh))

        self._needs_adapter = in_channels != out_channels
        if self._needs_adapter:
            P = self._build_isometric_projection(
                in_dim=in_channels,
                out_dim=out_channels,
                device=device,
            )                                      
            self.register_buffer("P", P)
            self.h0_adapter = lambda x, W=P: torch.matmul(x, W.T)
    
    @staticmethod
    def _build_isometric_projection(
        *, in_dim: int, out_dim: int, device: Optional[torch.device]
    ) -> torch.Tensor:
        if out_dim >= in_dim:
            G = torch.randn(out_dim, in_dim, device=device)
            Q, _ = torch.linalg.qr(G, mode="reduced")
            P = Q
        else:   
            G = torch.randn(in_dim, out_dim, device=device)
            Q, _ = torch.linalg.qr(G, mode="reduced")
            P = Q.T.contiguous()
        return P
        
        
            
    @torch.no_grad()
    def enforce_spectral_caps(self) -> None:
        if self._spectral_caps is None:
            return

        for mod, cap in ((self.phi_self, self._spectral_caps[0]),
                        (self.phi_neigh, self._spectral_caps[1])):

            if not hasattr(mod, "weight_orig"):
                continue

            sigma = torch.linalg.norm(mod.weight_orig, ord=2).item()

            if sigma > cap:
                mod.weight_orig.mul_(cap / sigma)
                
    # ------------------------------------------------------------------
    #  forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        *,
        initial_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        no_edge_weights = False
        num_nodes = x.size(0)
        device = x.device
        if edge_weight is None:
            no_edge_weights = True
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
        is_undirected = _is_undirected(edge_index, num_nodes)
        if not hasattr(self, "h0"):
            self.register_buffer("h0", initial_x if initial_x is not None else x.detach())
        elif initial_x is not None:
            self.h0.copy_(initial_x)

        # 0) Spectral caps
        self.enforce_spectral_caps()

        # 1) --- Ricci curvature of current metric -------------------------------------------------
        combinatorial_only = no_edge_weights and is_undirected
        kappa = lly_curvature_limit_free(edge_index, num_nodes, edge_weight, combinatorial_only=combinatorial_only)

        # 2) --- Explicit half‑Euler Ricci‑flow step and renormalization ---------------------------
        delta_t = cfl_delta_t(kappa, edge_weight)
        w_half = ricci_flow_half_step(edge_weight, kappa, delta_t)

        # 3) --- Metric surgery---- ----------------------------------------------------------------
        edge_index_new, w_half = metric_surgery(edge_index, w_half)

        # 4) --- Row‑normalise and Laplacian -------------------------------------------------------
        w_norm = row_normalise(edge_index_new, w_half, num_nodes)
        lap_vals = laplacian(edge_index_new, w_norm, num_nodes)  # = −w_norm
        minus_L = -lap_vals  # positive weights for neighbour aggregation

        # 5) --- Curvature gate --------------------------------------------------------------------
        mean_kappa = incident_curvature(edge_index_new, kappa, num_nodes)
        rho = curvature_gate(mean_kappa)  # shape (N,)

        # 6) --- tau‑budget ------------------------------------------------------------------------
        if not self._tau_fixed and self.tau.item() == 0.0:
            rho_max = float(rho.max().item())
            s_self, s_neigh, tau_val = allocate_tau_budget(rho_max, delta=self.delta, r=self.r)
            self.tau.fill_(tau_val)
            # If spectral caps were not given, adopt those implicit budgets
            if self._spectral_caps is None:
                self._spectral_caps = (s_self, s_neigh)
                # retrofit spectral_norm wrappers so that future calls to
                # enforce_spectral_caps() clip correctly
                self.phi_self = spectral_norm(self.phi_self)
                self.phi_neigh = spectral_norm(self.phi_neigh)
                self.register_buffer("_cap_self", torch.tensor(s_self))
                self.register_buffer("_cap_neigh", torch.tensor(s_neigh))
            # up to the caller to renormalise phi weights externally

        # 7) --- Linear projections -------------------------------------------
        h_self = self.phi_self(x)  # (N, d_out)
        h_neigh = self.phi_neigh(x)  # (N, d_out)

        # 8) --- Neighbour aggregation via −L_{vu} phi_neigh h_u^{(k)} ------------
        row, col = edge_index_new  # E' elements each
        msg = minus_L.unsqueeze(-1) * h_neigh[col]  # (E', d_out)
        if scatter_add is not None:
            neigh_aggr = scatter_add(msg, row, dim=0, dim_size=num_nodes)
        else:
            neigh_aggr = torch.zeros(num_nodes, self.out_channels, device=device, dtype=x.dtype)
            neigh_aggr.index_add_(0, row, msg)

        final_h0 = self.h0 if not self._needs_adapter else self.h0_adapter(self.h0) ##Projection if needed

        # 9) --- Residual & teleport ------------------------------------------
        out = rho.unsqueeze(-1) * h_self + neigh_aggr + self.tau * final_h0

        return out, edge_index_new, w_norm
