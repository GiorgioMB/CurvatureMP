# ============================================================
#  CGMP — Utilities, Jacobians and Energy Functionals
# ============================================================
import torch
import math
from scipy.optimize import linprog
from itertools import combinations
import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict

# ------------------------------------------------------------------
# 0.  Required utilities
# ------------------------------------------------------------------
def _is_undirected(edge_index: torch.LongTensor, num_nodes: int) -> bool:
    row, col = edge_index
    mask = row != col  # disregard self‑loops
    row = row[mask]
    col = col[mask]
    # Unique id for every directed edge
    idx = row * num_nodes + col
    idx_rev = col * num_nodes + row
    return torch.equal(torch.sort(idx).values, torch.sort(idx_rev).values)

def _dtype_bits(tensor: torch.Tensor) -> int:
    if tensor.is_floating_point():
        return torch.finfo(tensor.dtype).bits         
    else:                                              
        return torch.iinfo(tensor.dtype).bits

def _dijkstra_excluding(
    src: int,
    tgt: int,
    adj: List[Dict[int, float]],
    banned: Tuple[int, int],
) -> float:
    dist = [math.inf] * len(adj)
    dist[src] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue  # stale entry
        if u == tgt:
            return d  # early exit
        for v, w in adj[u].items():
            if (u, v) == banned or (v, u) == banned:
                continue  # skip the removed edge (both orientations)
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return math.inf  # unreachable

def _dijkstra_restricted(
    source: int,
    nodes_set: set[int],
    adj_weight: list[Dict[int, float]],
) -> Dict[int, float]:

    dist: Dict[int, float] = {source: 0.0}
    seen: set[int] = set()
    pq: list[Tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        for v, w in adj_weight[u].items():
            if v not in nodes_set:
                continue
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist


def _dfs_paths(layer_graphs: List[Dict[int, List[int]]],
               depth: int,
               cur: int,
               target: int,
               path: List[int],
               out: List[List[int]]):
    if depth == 0:
        if cur == target:
            out.append(path.copy())
        return
    for nxt in layer_graphs[len(path) - 1].get(cur, []):   # neighbours in this layer
        path.append(nxt)
        _dfs_paths(layer_graphs, depth - 1, nxt, target, path, out)
        path.pop()



def _hop_block(v_prev: int, v_cur: int, layer: int,
               rho: List[torch.Tensor],
               lap_vals: List[Dict[Tuple[int, int], float]],
               Phi_self: torch.Tensor,
               Phi_neigh: torch.Tensor) -> torch.Tensor:
    if v_cur == v_prev:
        return rho[layer][v_cur] * Phi_self
    else:
        return -lap_vals[layer][(v_cur, v_prev)] * Phi_neigh


def _J_hom(path: List[int],
           rho: List[torch.Tensor],
           lap_vals: List[Dict[Tuple[int, int], float]],
           Phi_self: torch.Tensor,
           Phi_neigh: torch.Tensor) -> torch.Tensor:
    J = torch.eye(Phi_self.shape[0], device=Phi_self.device)
    for hop in range(1, len(path)):                    
        layer = hop - 1                                
        B = _hop_block(path[hop - 1], path[hop],
                       layer, rho, lap_vals,
                       Phi_self, Phi_neigh)
        J = B @ J
    return J



def _J_tele(path_tail: List[int],     
            m: int,                 
            tau: float,
            rho: List[torch.Tensor],
            lap_vals: List[Dict[Tuple[int, int], float]],
            Phi_self: torch.Tensor,
            Phi_neigh: torch.Tensor) -> torch.Tensor:
    """Single-teleport Jacobian contribution."""
    if tau == 0.0:
        return torch.zeros_like(Phi_self)

    J = torch.eye(Phi_self.shape[0], device=Phi_self.device)
    for hop in range(1, len(path_tail)):             
        layer = m + hop                               
        B = _hop_block(path_tail[hop - 1], path_tail[hop],
                       layer, rho, lap_vals,
                       Phi_self, Phi_neigh)
        J = B @ J
    return tau * J



# ------------------------------------------------------------------
# 1.  LLY curvature, Ricci flow, metric surgery helpers
# ------------------------------------------------------------------
def lly_curvature_limit_free(
    edge_index: torch.LongTensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    *,
    combinatorial_only: bool = True,
) -> torch.Tensor:
    if edge_weight is not None:
        dtype = edge_weight.dtype
    else:
        dtype = torch.get_default_dtype()

    # -----------------------------------------------------------------------
    # Build undirected adjacency with weights 
    # -----------------------------------------------------------------------
    row, col = edge_index[0].tolist(), edge_index[1].tolist()
    if edge_weight is None:
        lengths = torch.ones(len(row), dtype=dtype, device=edge_index.device)
    else:
        lengths = edge_weight.to(dtype)

    adj_weight: list[Dict[int, float]] = [dict() for _ in range(num_nodes)]
    weight_uv: Dict[Tuple[int, int], float] = {}

    for u, v, w in zip(row, col, lengths.cpu().tolist()):
        if u == v:
            continue  # ignore self‑loops
        # keep the smallest weight if duplicates exist
        if w < adj_weight[u].get(v, float("inf")):
            adj_weight[u][v] = adj_weight[v][u] = w
            weight_uv[(u, v)] = weight_uv[(v, u)] = w

    # -----------------------------------------------------------------------
    # Degrees 
    # -----------------------------------------------------------------------
    if edge_weight is None:
        deg_vals = [len(adj_weight[v]) for v in range(num_nodes)]
    else:
        deg_vals = [float(sum(adj_weight[v].values())) for v in range(num_nodes)]
    deg = torch.tensor(deg_vals, dtype=dtype, device=edge_index.device)

    # -----------------------------------------------------------------------
    # Helper: combinatorial closed form (unit lengths only)
    # -----------------------------------------------------------------------
    def _closed_form(u: int, v: int) -> float:
        Su, Sv = set(adj_weight[u].keys()), set(adj_weight[v].keys())
        common = Su & Sv
        T = len(common)  # triangles
        only_u, only_v = Su - Sv, Sv - Su
        Sq = sum(1 for x in only_u for y in only_v if y in adj_weight[x])  # squares
        return float(2 + T - Sq - (len(only_u) + len(only_v) - Sq))

    # -----------------------------------------------------------------------
    # Main loop over directed edges
    # -----------------------------------------------------------------------
    E = edge_index.size(1)
    kappa = torch.empty(E, dtype=dtype, device=edge_index.device)

    for idx in range(E):
        u = edge_index[0, idx].item()
        v = edge_index[1, idx].item()

        # Fast path: unweighted closed form
        if combinatorial_only:
            kappa[idx] = _closed_form(u, v)
            continue

        
        Nu = [u] + list(adj_weight[u].keys())
        Nv = [v] + list(adj_weight[v].keys())
        nodes = list({*Nu, *Nv})
        n = len(nodes)
        idx_of = {w: i for i, w in enumerate(nodes)}
        nodes_set = set(nodes)

        sp_cache: Dict[int, Dict[int, float]] = {
            a: _dijkstra_restricted(a, nodes_set, adj_weight) for a in nodes
        }

        A_eq, b_eq, A_ub, b_ub = [], [], [], []

        d_uv = weight_uv[(u, v)] if (u, v) in weight_uv else 1.0
        eq = np.zeros(n)
        eq[idx_of[v]], eq[idx_of[u]] = +1.0, -1.0
        A_eq.append(eq)
        b_eq.append(d_uv)

        for a, b in combinations(nodes, 2):
            dist_ab = sp_cache[a].get(b, float("inf"))
            if not np.isfinite(dist_ab):
                continue
            c1, c2 = np.zeros(n), np.zeros(n)
            c1[idx_of[a]], c1[idx_of[b]] = +1, -1
            c2[idx_of[a]], c2[idx_of[b]] = -1, +1
            A_ub.extend([c1, c2])
            b_ub.extend([dist_ab, dist_ab])

        c_obj = np.zeros(n)
        for nb, w in adj_weight[u].items():
            c_obj[idx_of[u]] -= w / deg_vals[u]
            c_obj[idx_of[nb]] += w / deg_vals[u]
        for nb, w in adj_weight[v].items():
            c_obj[idx_of[v]] += w / deg_vals[v]
            c_obj[idx_of[nb]] -= w / deg_vals[v]
        if not np.all(np.isfinite(c_obj)):
            print("bad c_obj", idx, u, v)
        if A_ub is not None and not np.all(np.isfinite(A_ub)):
            print("bad A_ub")
        if b_ub is not None and not np.all(np.isfinite(b_ub)):
            print("bad b_ub")
        if not np.all(np.isfinite(A_eq)) or not np.all(np.isfinite(b_eq)):
            print("bad A_eq/b_eq")
        dist_uv = sp_cache[u].get(v, float("inf"))
        if dist_uv < d_uv - 1e-10:        # strict enough tolerance
            print(f"Infeasible edge: ({u},{v})  w={d_uv}  dist={dist_uv}")

        res = linprog(
            c=c_obj,
            A_ub=np.asarray(A_ub) if A_ub else None,
            b_ub=np.asarray(b_ub) if b_ub else None,
            A_eq=np.asarray(A_eq),
            b_eq=np.asarray(b_eq),
            bounds=(None, None),
            method="highs",
        )
        if not res.success:
            print(res.status, res.message)

        kappa[idx] = res.fun if res.success else float("nan")

    return kappa


def cfl_delta_t(curvature: torch.Tensor, edge_weight: torch.Tensor) -> float:
    max_k = curvature.abs().max() 
    if max_k == 0:
        return 1.0

    tot_w = edge_weight.sum()
    bits  = _dtype_bits(edge_weight)
    delta_t = 1.0 / (bits * max_k * (1.0 + tot_w))
    return float(delta_t.item()) 


# ---------------------------------------------------------------------------
# One half Ricci‑flow step
# ---------------------------------------------------------------------------

def ricci_flow_half_step(
    edge_weight: torch.Tensor,
    curvature: torch.Tensor,
    delta_t: Optional[float] = None,
) -> torch.Tensor:

    if delta_t is None:
        delta_t = cfl_delta_t(curvature, edge_weight)
    dt = torch.as_tensor(delta_t, dtype=edge_weight.dtype, device=edge_weight.device)

    S = (curvature * edge_weight).sum()  # scalar in same dtype/device
    new_w = edge_weight * (1.0 - dt * (curvature - S))

    tiny = torch.finfo(edge_weight.dtype).tiny
    scale = edge_weight.sum() / new_w.sum().clamp_min(tiny)
    return new_w * scale

def metric_surgery(
    edge_index: torch.LongTensor,
    edge_weight: torch.Tensor,
    is_undirected: bool = False,
    verbose: bool = False,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    
    device = edge_weight.device
    dtype = edge_weight.dtype

    row, col = edge_index  # (2, E)
    E = int(edge_weight.numel())
    if E == 0:
        if verbose:
            print("[metric_surgery] Empty edge set; nothing to do.")
        return edge_index, edge_weight

    n = int(torch.max(edge_index).item()) + 1  

    adj: List[Dict[int, float]] = [dict() for _ in range(n)]
    row_l = row.tolist()
    col_l = col.tolist()
    w_l   = edge_weight.tolist()
    for u, v, w in zip(row_l, col_l, w_l):
        adj[u][v] = float(w)

    
    pair_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for i, (u, v) in enumerate(zip(row_l, col_l)):
        pair_to_indices.setdefault((u, v), []).append(i)

    keep = torch.ones(E, dtype=torch.bool, device=device)
    removed_primary = 0
    tol_scale = 1e-14  
    for idx in range(E):
        u = row_l[idx]
        v = col_l[idx]
        w = float(w_l[idx])

        d_uv = _dijkstra_excluding(u, v, adj, banned=(u, v))

        if not math.isfinite(d_uv):
            continue

        if w > d_uv + tol_scale * max(1.0, abs(w)):
            keep[idx] = False
            removed_primary += 1

    extra_sym_removed = 0
    if is_undirected and removed_primary > 0:
        to_drop_extra: List[int] = []
        for idx in range(E):
            if keep[idx]:
                continue
            u = row_l[idx]
            v = col_l[idx]
            rev_indices = pair_to_indices.get((v, u), [])
            for j in rev_indices:
                if keep[j]:
                    keep[j] = False
                    extra_sym_removed += 1
                    to_drop_extra.append(j)

    edge_index_out = edge_index[:, keep]
    edge_weight_out = edge_weight[keep]

    if verbose:
        total_removed = int((~keep).sum().item())
        msg = f"[metric_surgery] Removed {total_removed} edges"
        if is_undirected:
            msg += f" ({removed_primary} primary + {extra_sym_removed} symmetric)"
        print(msg)

    return edge_index_out, edge_weight_out



# ------------------------------------------------------------------
# 2.  Normalisation, Laplacian, curvature gate
# ------------------------------------------------------------------
def row_normalise(edge_index: torch.LongTensor,
                  edge_weight: torch.Tensor,
                  num_nodes: int
                  ) -> torch.Tensor:
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=edge_weight.device).scatter_add_(
        0, row, edge_weight)
    norm_w = edge_weight / deg[row].clamp_min(1e-18) # W dot
    return norm_w


def laplacian(edge_index: torch.LongTensor,
              row_norm_weight: torch.Tensor,
              num_nodes: int
              ) -> torch.Tensor:
    return -row_norm_weight #L^(k) 


def incident_curvature(edge_index: torch.LongTensor,
                       curvature: torch.Tensor,
                       num_nodes: int
                       ) -> torch.Tensor:
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=curvature.device).scatter_add_(
        0, row, torch.ones_like(curvature))
    accum = torch.zeros(num_nodes, device=curvature.device).scatter_add_(
        0, row, curvature)
    return accum / deg.clamp_min(1) # Equation (11.3)


def curvature_gate(mean_kappa: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.sigmoid(mean_kappa) # rho^(k+1)_v



# -------------------------------------------------------------
# 3.  tau-budget allocator 
# -------------------------------------------------------------

def allocate_tau_budget(rho_max: float,
                        delta: float = 0.10,
                        r: float = 2.0,
                        eps_root: float = 1e-6
                       ) -> Tuple[float, float, float]:

    def tau_min(gamma: float) -> float:
        return gamma * (1.0 - gamma) / (1.0 - 2.0 * gamma)

    def root_solve(delta_: float) -> float:
        a, b = 0.0, 0.5 - eps_root    
        while b - a > eps_root:
            gamma = 0.5 * (a + b)
            if tau_min(gamma) + gamma < 1.0 - delta_:
                a = gamma
            else:
                b = gamma
        return 0.5 * (a + b)           
    
    gamma_star = root_solve(delta)


    s_neigh = gamma_star / (rho_max * r + 2.0)
    s_self  = r * s_neigh

    tau_min_star = tau_min(gamma_star)
    eps_slack    = 0.05 * (1.0 - delta - gamma_star)
    tau          = tau_min_star + eps_slack

    return s_self, s_neigh, tau



# ------------------------------------------------------------------
# 4.  Jacobians
# ------------------------------------------------------------------
def layer_jacobian_sparse(
        edge_index:      torch.LongTensor,   
        laplacian_vals:  torch.Tensor,       
        rho:             torch.Tensor,      
        Phi_self:        torch.Tensor,    
        Phi_neigh:       torch.Tensor      
    ) -> torch.sparse_coo_tensor:
    device  = rho.device
    n       = rho.numel()
    d_out, d_in = Phi_self.shape                    
    block_elems = d_out * d_in

    row_pattern = torch.arange(d_out, device=device).repeat_interleave(d_in) 
    col_pattern = torch.arange(d_in,  device=device).repeat(d_out)          

    diag_rows = row_pattern.repeat(n) + torch.repeat_interleave(
                   torch.arange(n, device=device) * d_out, block_elems)
    diag_cols = col_pattern.repeat(n) + torch.repeat_interleave(
                   torch.arange(n, device=device) * d_in, block_elems)

    diag_vals = torch.kron(rho, Phi_self.flatten())

    v_idx, u_idx = edge_index           
    E            = v_idx.numel()

    off_rows = row_pattern.repeat(E) + torch.repeat_interleave(
                 v_idx * d_out, block_elems)
    off_cols = col_pattern.repeat(E) + torch.repeat_interleave(
                 u_idx * d_in,  block_elems)

    off_vals = (-laplacian_vals).repeat_interleave(block_elems) \
               * Phi_neigh.flatten().repeat(E)

    rows = torch.cat([diag_rows, off_rows])
    cols = torch.cat([diag_cols, off_cols])
    vals = torch.cat([diag_vals, off_vals])

    shape = (n * d_out, n * d_in)
    return torch.sparse_coo_tensor(torch.stack([rows, cols]),
                                   vals,
                                   size=shape,
                                   device=device).coalesce()


def depth_jacobian(layer_ops: List[torch.Tensor],
                   tau: float = 0.0
                   ) -> torch.Tensor:
    if not layer_ops:
        raise ValueError("Need at least one layer operator")
    product = torch.eye(layer_ops[0].shape[0], device=layer_ops[0].device)
    for T in layer_ops:
        product = T @ product
    if tau == 0.0:
        return product
    # inhomogeneous term
    L = len(layer_ops)
    sums = torch.zeros_like(product)
    right_prod = torch.eye(product.shape[0], device=product.device)
    for m in range(L - 1, -1, -1):
        sums += right_prod
        right_prod = layer_ops[m] @ right_prod
    return product + tau * sums

## Warning: extremely slow for large L
def pathwise_jacobian(u: int,
                      v: int,
                      L: int,
                      layer_graphs: List[Dict[int, List[int]]],
                      rho: List[torch.Tensor],
                      lap_vals: List[Dict[Tuple[int, int], float]],
                      Phi_self: torch.Tensor,
                      Phi_neigh: torch.Tensor,
                      tau: float = 0.0) -> torch.Tensor:
    device = Phi_self.device
    d = Phi_self.shape[0]
    J_total = torch.zeros((d, d), device=device)

    # ---- homogeneous term ----
    hom_paths: List[List[int]] = []
    _dfs_paths(layer_graphs, L, u, v, [u], hom_paths)
    for p in hom_paths:
        J_total += _J_hom(p, rho, lap_vals, Phi_self, Phi_neigh)

    # ---- one-teleport terms ----
    for m in range(0, L):                          
        tail_depth = L - m - 1
        layer_graphs_tail = layer_graphs[m + 1:]  
        tail_paths: List[List[int]] = []
        _dfs_paths(layer_graphs_tail, tail_depth, u, v, [u], tail_paths)
        for q in tail_paths:
            J_total += _J_tele(q, m, tau, rho, lap_vals, Phi_self, Phi_neigh)

    return J_total


def pathwise_amplification(path: List[int],
                           rho: torch.Tensor,
                           lap_vals: Dict[Tuple[int, int], float],
                           Phi_self: torch.Tensor,
                           Phi_neigh: torch.Tensor
                           ) -> float:
    gamma = 1.0
    for i in range(1, len(path)):
        v_prev, v_cur = path[i - 1], path[i]
        if v_cur == v_prev:      # self hop
            gamma *= rho[v_cur].item() * torch.linalg.norm(Phi_self, 2).item()
        else:                    # neighbour hop
            L_abs = abs(lap_vals[(v_cur, v_prev)])
            gamma *= L_abs * torch.linalg.norm(Phi_neigh, 2).item()
    return gamma



# ------------------------------------------------------------------
# 5.  Energy functionals
# ------------------------------------------------------------------
def dirichlet_energy(h, edge_index, edge_weight, assume_undirected: bool | None = None):
    if assume_undirected is None:   # auto-detect once
        assume_undirected = _is_undirected(edge_index, h.size(0))

    row, col = edge_index
    diff2 = (h[row] - h[col]).square().sum(dim=-1)
    coeff = 0.5 if assume_undirected else 1.0
    return coeff * (edge_weight * diff2).sum()



def curvature_variance_energy(curvature: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    S = (curvature * edge_weight).sum()
    return 0.5 * torch.sum(edge_weight * (curvature - S).pow(2))

def oversquashing_index(
    jac_stack: torch.Tensor,   # (L, N*d_out, N*d_in)
    dist: torch.Tensor,        # (N, N)
    eta: float = 0.0           # threshold
) -> float:
    if jac_stack.numel() == 0:
        return 1.0
    jac_stack = jac_stack.double()
    # Ensure dist lives on the same device as jac_stack
    dist = dist.to(jac_stack.device)

    L, Nd_out, Nd_in = jac_stack.shape
    N = dist.size(0)
    d_out = Nd_out // N
    d_in  = Nd_in  // N
    assert Nd_out == N * d_out and Nd_in == N * d_in, "bad Jacobian shape"

    # (L, N, d_out, N, d_in)
    J = jac_stack.view(L, N, d_out, N, d_in)

    # 2-norm of each (d_out × d_in) block  ->  (L, N, N)
    block_norm = torch.linalg.norm(J, ord=2, dim=(2, 4))

    # Flatten node pairs once
    d_flat = dist.flatten()                  # (N^2,)
    bn_flat = block_norm.view(L, -1)         # (L, N^2)

    # Last depth
    mask = bn_flat.max(dim=0).values > eta
    if not mask.any():
        return 1.0          # index = 1  (no block above threshold)

    last_good = int(d_flat[mask].max())
    return 1.0 / (1.0 + last_good)
