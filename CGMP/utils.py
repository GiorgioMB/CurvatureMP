# ============================================================
#  CGMP — Utilities, Jacobians and Energy Functionals
# ============================================================
import torch
import math
from typing import List, Tuple, Optional, Dict

# ------------------------------------------------------------------
# 0.  Required curvature utilities
# ------------------------------------------------------------------
def _logsumexp(M, dim):
    m = M.max(dim=dim, keepdim=True).values
    return m + torch.log(torch.exp(M - m).sum(dim=dim, keepdim=True))


def _sinkhorn_cost_log(
    C, 
    a,
    b, 
    eps: float = 1e-16,
    iters: int = 300,
    tol: float = 1e-16,
    n_scale: int = 4):
    
    device, dtype = C.device, C.dtype
    log_a = torch.log(a + 1e-38)
    log_b = torch.log(b + 1e-38)

    K_log = -C / eps  
    log_u = torch.zeros_like(a) 
    log_v = torch.zeros_like(b)

    for _ in range(n_scale):
        for _ in range(iters):
            log_u_prev = log_u.clone()
            log_v = (
                log_b - _logsumexp(K_log.T + log_u[None, :], dim=1).squeeze(1)
            )
            log_u = (
                log_a - _logsumexp(K_log + log_v[None, :], dim=1).squeeze(1)
            )
            if torch.max(torch.abs(log_u - log_u_prev)) < tol:
                break
        eps *= 0.5
        K_log = -C / eps

    log_Pi = log_u[:, None] + K_log + log_v[None, :]
    Pi = torch.exp(log_Pi) 
    m, n = a.numel(), b.numel()
    Piaa = a[:, None] / m  
    Pibb = b[None, :] / n 
    return (Pi * C).sum() - 0.5 * ((Piaa * C).sum() + (Pibb * C).sum())


def _bellman_ford(
        src: int, 
        edges: List[Tuple[int, int, float]], 
        n: int) -> torch.Tensor:
    dist = torch.full((n,), math.inf, dtype=torch.float32)
    dist[src] = 0.0
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                updated = True
        if not updated:
            break
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


def _compute_p_curvature(
        p, 
        u, 
        v, 
        d_uv, 
        adj_len, 
        cache, 
        eps, 
        iters, 
        device, 
        Nu_ext, 
        Nv_ext
    ) -> float:
    # masses: [p] on the root, (1-p)*(length/deg) on neighbours
    a_vals = [p] + [(1 - p) * (l / sum(adj_len[u])) for l in adj_len[u]]
    b_vals = [p] + [(1 - p) * (l / sum(adj_len[v])) for l in adj_len[v]]
    a_p = torch.tensor(a_vals, dtype=torch.float32, device=device)
    b_p = torch.tensor(b_vals, dtype=torch.float32, device=device)

    # cost matrix of size (|Nu_ext| times |Nv_ext|)
    C_p = torch.empty((len(Nu_ext), len(Nv_ext)), device=device)
    for i, uu in enumerate(Nu_ext):
        # distances from uu to every node in Nv_ext
        C_p[i] = cache[uu][Nv_ext]

    W_p = _sinkhorn_cost_log(C_p, a_p, b_p, eps, iters)
    return 1.0 - W_p / d_uv


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



def compute_LLY_curvature(
    edge_index: torch.LongTensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-16,
    iters: int = 300,
    dp: float = 1e-6) -> torch.Tensor:

    row, col = edge_index
    device = edge_index.device
    lengths = (
        torch.ones_like(row, dtype=torch.float32)
        if edge_weight is None
        else edge_weight.float()
    )

    # Create adjacency lists and lengths
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    adj_len: List[List[float]] = [[] for _ in range(num_nodes)]
    edges_for_bf: List[Tuple[int, int, float]] = []

    for u, v, w in zip(row.tolist(), col.tolist(), lengths.tolist()):
        adj[u].append(v)
        adj[v].append(u)
        adj_len[u].append(w)
        adj_len[v].append(w)

        edges_for_bf.append((u, v, w))
        edges_for_bf.append((v, u, w))

    E = row.numel()
    kappa = torch.empty(E, dtype=torch.float32, device=device)
    cache = {}
    for e in range(E):
        u, v = row[e].item(), col[e].item()
        d_uv = lengths[e].item()

        if len(adj[u]) <= 1 or len(adj[v]) <= 1:
            kappa[e] = 1.0
            continue
        # Compute p-curvature for p1 and p2
        p1 = 1.0 - dp
        p2 = 1.0 - 2*dp
        Nu_ext = [u] + adj[u]
        Nv_ext = [v] + adj[v]

        if u not in cache:
            cache[u] = _bellman_ford(u, edges_for_bf, num_nodes).to(device)
        for uu in adj[u]:
            if uu not in cache:
                cache[uu] = _bellman_ford(uu, edges_for_bf, num_nodes).to(device)
        # compute p-slack Ollivier Ricci Curvature with slack p1 and p2
        k1 = _compute_p_curvature(p1, u, v, d_uv, adj_len, cache, eps, iters, device, Nu_ext, Nv_ext)
        k2 = _compute_p_curvature(p2, u, v, d_uv, adj_len, cache, eps, iters, device, Nu_ext, Nv_ext)
        kappa[e] = (k1 - k2) / dp # Finite difference approximation
    return kappa



# ------------------------------------------------------------------
# 1.  Ricci–flow helpers
# ------------------------------------------------------------------
def cfl_delta_t(curvature: torch.Tensor,
                edge_weight: torch.Tensor) -> float:
    max_k = curvature.abs().max().item() ##Infinity norm
    tot_w = edge_weight.sum().item() # s^(k)
    return 1.0 if max_k == 0 else 1.0 / (max_k * (1.0 + tot_w)) #Eq. (11.2)


def ricci_flow_half_step(edge_weight: torch.Tensor,
                         curvature: torch.Tensor,
                         delta_t: Optional[float] = None
                         ) -> torch.Tensor:
    if delta_t is None:
        delta_t = cfl_delta_t(curvature, edge_weight)
    S = (curvature * edge_weight).sum() # S^(k)
    new_w = edge_weight * (1.0 - delta_t * (curvature - S)) #Eq. (11.1)
    scale = edge_weight.sum() / new_w.sum().clamp_min(1e-18) #s^(k) / s^(k+(1/2))
    return new_w * scale # w dot


def metric_surgery(edge_index: torch.LongTensor,
                   edge_weight: torch.Tensor
                   ) -> Tuple[torch.LongTensor, torch.Tensor]:
    row, col = edge_index
    device = edge_weight.device
    n = int(row.max().item()) + 1
    keep = torch.ones_like(edge_weight, dtype=torch.bool)

    edges = [(u.item(), v.item(), w.item())
             for u, v, w in zip(row, col, edge_weight)]

    for e, (u, v, w) in enumerate(zip(row.tolist(),
                                      col.tolist(),
                                      edge_weight.tolist())):
        minus = [e_ for i, e_ in enumerate(edges) if i != e and
                 not (e_[0] == v and e_[1] == u)]
        d_uv = _bellman_ford(u, minus, n)[v].item()
        keep[e] = w <= d_uv # Exit Condition (I)
    return edge_index[:, keep], edge_weight[keep]



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
def dirichlet_energy(h: torch.Tensor,
                     edge_index: torch.LongTensor,
                     edge_weight: torch.Tensor
                     ) -> torch.Tensor:
    row, col = edge_index
    diff = h[row] - h[col]
    return 0.5 * torch.sum(edge_weight * diff.pow(2)) #Equation (11.8)


def curvature_variance_energy(curvature: torch.Tensor,
                              edge_weight: torch.Tensor
                              ) -> torch.Tensor:
    S = (curvature * edge_weight).sum()
    return 0.5 * torch.sum(edge_weight * (curvature - S).pow(2)) #Equation (11.9)


def oversquashing_index(
        depth_jacobian: torch.Tensor,   # shape: (D, N_pairs)
        eta: float = 1e-2,
        assume_unbounded: bool = False  
    ) -> float:

    if depth_jacobian.numel() == 0:
        return 1.0

    per_depth_max = depth_jacobian.norm(p=2, dim=1)  
    above = (per_depth_max > eta).nonzero(as_tuple=True)[0]

    if len(above) == 0:
        return 1.0

    last_good_depth = int(above.max().item())   
    if (last_good_depth == depth_jacobian.shape[0]-1) and assume_unbounded:
        # We cannot see beyond depth D, but the last observed depth is still > eta,
        # so we tentatively treat the supremum as +infinity (index = 0.0).
        return 0.0

    # Finite, non-empty supremum
    return 1.0 / (1.0 + last_good_depth) # Equation (11.10)
