import torch, math
from typing import List, Tuple, Optional

def _logsumexp(M, dim):
    m = M.max(dim=dim, keepdim=True).values
    return m + torch.log(torch.exp(M - m).sum(dim=dim, keepdim=True))

# ----------------------------
# Sinkhorn divergence
# ----------------------------
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


# -----------------------
# Bellman–Ford algorithm
# -----------------------
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

# ---------------------------------------
# Curvature computation for some slack p
# ---------------------------------------
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

# ----------------------------------------------
#  Lin–Lu–Yau curvature (stochastic‑matrix form)
# ----------------------------------------------
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

    # Build weighted adjacency for both random walk and distances
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

        # Pendant edge -> kappa = 1 by definition (optional: drop this if exact value wanted)
        if len(adj[u]) <= 1 or len(adj[v]) <= 1:
            kappa[e] = 1.0
            continue


        p1 = 1.0 - dp
        p2 = 1.0 - 2*dp
        Nu_ext = [u] + adj[u]
        Nv_ext = [v] + adj[v]

        if u not in cache:
            cache[u] = _bellman_ford(u, edges_for_bf, num_nodes).to(device)
        for uu in adj[u]:
            if uu not in cache:
                cache[uu] = _bellman_ford(uu, edges_for_bf, num_nodes).to(device)
        k1 = _compute_p_curvature(p1, u, v, d_uv, adj_len, cache, eps, iters, device, Nu_ext, Nv_ext)
        k2 = _compute_p_curvature(p2, u, v, d_uv, adj_len, cache, eps, iters, device, Nu_ext, Nv_ext)
        kappa[e] = (k1 - k2) / dp # Finite difference approximation
    return kappa


# ---------------------------------------
# One step of Ollivier‑Ricci
# ---------------------------------------
def compute_ORF_step(
    edge_index: torch.LongTensor,
    curvature: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    delta_t: Optional[float] = None,
    normalised: bool = True):

    row, col = edge_index
    device = curvature.device

    w = torch.ones_like(curvature) if edge_weight is None else edge_weight.clone()

    # CFL adaptive step
    if delta_t is None:
        max_k = torch.norm(curvature, p=float("inf")).item()
        inv_delta = 2 * (max_k if max_k != 0 else 1.0)
        step = 1 / (inv_delta)
        print(f"Time-step: {step:.4f} (max curvature: {max_k:.4f})", end='\r')

    else:
        step = delta_t

    # ODE step
    if normalised:
        Lambda = (curvature * w).sum()
        new_w = w + step * w * (-curvature + Lambda)
        # Enforce exact invariance by rescaling
        new_w *= w.sum() / new_w.sum() 
    else:
        new_w = w - step * curvature * w

    n_nodes   = int(row.max().item()) + 1
    all_edges = [(u, v, wt) for u, v, wt in
                 zip(row.tolist(), col.tolist(), new_w.tolist())]
    keep_mask = torch.ones_like(new_w, dtype=torch.bool)

    for idx, (u, v, wt) in enumerate(zip(row.tolist(), col.tolist(), new_w)):
        minus = [e for e in all_edges if not ((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u))]
        dist_uv = _bellman_ford(u, minus, n_nodes)[v].item()

        keep_mask[idx] = wt <= dist_uv
    return edge_index[:, keep_mask], new_w[keep_mask]
