import math
import torch
import types
import pytest
from ..CGMP.utils import (
    lly_curvature_limit_free,
    cfl_delta_t, ricci_flow_half_step, metric_surgery,
    row_normalise, laplacian, incident_curvature, curvature_gate,
    dirichlet_energy, curvature_variance_energy, oversquashing_index,
    layer_jacobian_sparse, depth_jacobian, pathwise_amplification,
    allocate_tau_budget, pathwise_jacobian, _dtype_bits, _dijkstra_excluding,
     _dijkstra_restricted, _dfs_paths, _hop_block, _J_hom, _J_tele
)

# ----------------------------------------------------------------------
#  Section 0  ──  Low-level utilities
# ----------------------------------------------------------------------
def test_dtype_bits_float_and_int():
    f32 = torch.tensor([0.],  dtype=torch.float32)
    i8  = torch.tensor([1],   dtype=torch.int8)
    assert _dtype_bits(f32) == 32
    assert _dtype_bits(i8)  == 8


def build_layer_ops(rho_list, lap_list, Phi_self, Phi_neigh):
    n = rho_list[0].numel()
    d_out, d_in = Phi_self.shape
    layer_ops = []
    for rho, lap in zip(rho_list, lap_list):
        blocks = []
        for v in range(n):
            row_blocks = []
            for u in range(n):
                if v == u:                               
                    block = rho[v] * Phi_self
                elif (v, u) in lap:                        
                    block = -lap[(v, u)] * Phi_neigh
                else:                                        
                    block = torch.zeros_like(Phi_self)
                row_blocks.append(block)
            blocks.append(torch.cat(row_blocks, dim=1))
        layer_ops.append(torch.cat(blocks, dim=0))
    return layer_ops


def extract_block(big_mat, v, u, d_out, d_in):
    r = slice(v * d_out, (v + 1) * d_out)
    c = slice(u * d_in, (u + 1) * d_in)
    return big_mat[r, c]

# ----------------------------------------------------------------------
#  Section 1  ──  Curvature & Ricci flow
# ----------------------------------------------------------------------

def _tiny_triangle():
    #      0
    #     / \
    #    1---2        all edge weights = 1
    edge_index = torch.tensor([[0, 1, 2],
                               [1, 2, 0]])
    w = torch.ones(edge_index.size(1))
    return edge_index, w

@pytest.fixture(name="tiny_adj")
def _tiny_adj():
    # 0──1──2   (all 1) plus a long chord 0──2 weight 10
    adj = [dict() for _ in range(3)]
    adj[0][1] = adj[1][0] = 1.0
    adj[1][2] = adj[2][1] = 1.0
    adj[0][2] = adj[2][0] = 10.0
    return adj

def test_dijkstra_excluding_skips_banned(tiny_adj):
    adj = tiny_adj
    # Without the 0-1 edge the cheapest 0->2 route is the long chord (10.0)
    d = _dijkstra_excluding(0, 2, adj, banned=(0, 1))
    assert math.isclose(d, 10.0, rel_tol=1e-9)

def test_dijkstra_restricted_subset(tiny_adj):
    adj = tiny_adj
    dist = _dijkstra_restricted(0, {0, 1}, adj)
    assert dist == {0: 0.0, 1: 1.0}                # node 2 is ignored

def test_dfs_paths_single_route():
    # Two-layer toy graph: 0->1->2 is the only length-2 path 0->2
    layer_graphs = [
        {0: [1]},             # layer 0
        {1: [2]}              # layer 1
    ]
    out: list[list[int]] = []
    _dfs_paths(layer_graphs, depth=2, cur=0, target=2, path=[0], out=out)
    assert out == [[0, 1, 2]]

def test_LLY_curvature_shape_and_range():
    edge_index, w = _tiny_triangle()
    kappa = lly_curvature_limit_free(edge_index, num_nodes=3, edge_weight=w)
    expected = torch.full((3,), 1.5, dtype=kappa.dtype)
    assert torch.allclose(kappa, expected, atol=1e-6)

def test_cfl_delta_t_formula():
    curv = torch.tensor([2.0, -1.0])
    w    = torch.tensor([1.0, 2.0])
    expected = 1.0 / (32 * 2.0 * (1.0 + 3.0))   # 1 / 256
    assert math.isclose(cfl_delta_t(curv, w), expected, rel_tol=1e-6)

def test_ricci_flow_conserves_total_metric():
    _, w = _tiny_triangle()
    kappa = torch.zeros_like(w)
    w_new = ricci_flow_half_step(w, kappa)
    assert torch.allclose(w_new.sum(), w.sum(), atol=1e-6)

def test_metric_surgery_removes_long_edge():
    # 0──1──2  (all 1), plus a long chord 0──2 weight 3
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                               [1, 0, 2, 1, 2, 0]])
    w = torch.tensor([1.,1.,1.,1.,3.,3.])
    new_idx, new_w = metric_surgery(edge_index, w)
    # the (0,2) chord should disappear
    assert torch.all(new_w < 2.5) and new_idx.size(1) == 4

# ----------------------------------------------------------------------
#  Section 2  ──  Normalisation & Laplacian
# ----------------------------------------------------------------------

def test_row_normalise_rowsum_one():
    edge_index, w = _tiny_triangle()
    norm = row_normalise(edge_index, w, num_nodes=3)
    row = edge_index[0]
    sums = torch.zeros(3).scatter_add_(0, row, norm)
    assert torch.allclose(sums, torch.ones(3))

def test_laplacian_is_negative_norm_weights():
    edge_index, w = _tiny_triangle()
    norm = row_normalise(edge_index, w, 3)
    L = laplacian(edge_index, norm, 3)
    assert torch.allclose(L, -norm)

def test_incident_curvature_node_average():
    edge_index = torch.tensor([[0, 1, 2, 1, 2, 0],
                               [1, 2, 0, 0, 1, 2]])
    curv = torch.tensor([1., 2., 3., 1., 2., 3.])  
    avg = incident_curvature(edge_index, curv, num_nodes=3)
    assert pytest.approx(avg[0].item(), rel=1e-6) == 2.0

def test_curvature_gate_maps_to_unit_interval():
    val = curvature_gate(torch.tensor([-10., 0., 10.]))
    assert torch.all((0.0 <= val) & (val <= 1.0))

# ----------------------------------------------------------------------
#  Section 3  ──  Energy functionals
# ----------------------------------------------------------------------

def test_dirichlet_energy_two_nodes():
    h = torch.tensor([0., 1.])
    edge_index = torch.tensor([[0], [1]])
    w = torch.tensor([1.])
    assert math.isclose(dirichlet_energy(h, edge_index, w).item(), 0.5)

def test_curvature_variance_matches_formula():
    curv = torch.tensor([0.2, 0.7, -0.1])
    w    = torch.tensor([1.0, 2.0, 1.0])
    S    = (curv * w).sum()             
    expected = 0.5 * torch.sum(w * (curv - S)**2)
    assert torch.isclose(curvature_variance_energy(curv, w), expected)

def test_oversquashing_index_all_zero():
    J = torch.zeros(4, 3)                  
    assert oversquashing_index(J) == 1.0

def test_oversquashing_index_last_depth_active():
    J = torch.zeros(4, 3)
    J[-1, 0] = 2 * 1e-2                      
    assert math.isclose(oversquashing_index(J), 1.0 / (1 + 3), rel_tol=1e-6)

def test_oversquashing_index_unbounded_branch():
    J = torch.zeros(4, 3)
    J[-1, 0] = 2 * 1e-2
    assert oversquashing_index(J, assume_unbounded=True) == 0.0

# ----------------------------------------------------------------------
#  Section 4  ──  Jacobians
# ----------------------------------------------------------------------

def _toy_layer_jacobian():
    edge_index = torch.tensor([[0, 1],
                               [1, 0]])
    L_vals = torch.tensor([-1., -1.])
    rho = torch.tensor([0.5, 0.5])
    Phi_self  = torch.eye(2)
    Phi_neigh = torch.eye(2)
    return layer_jacobian_sparse(edge_index, L_vals, rho,
                                 Phi_self, Phi_neigh)

def _mini_blocks():
    Phi_self  = torch.tensor([[2.0]])
    Phi_neigh = torch.tensor([[1.0]])
    rho       = [torch.tensor([0.5, 0.7]), torch.tensor([0.6, 0.8])]
    lap_vals  = [
        {(1, 0): 0.3, (0, 1): 0.3},
        {(1, 0): 0.4, (0, 1): 0.4},
    ]
    return rho, lap_vals, Phi_self, Phi_neigh

def test_hop_block_self_vs_neigh():
    rho, lap, phi_s, phi_n = _mini_blocks()
    self_blk = _hop_block(0, 0, layer=0, rho=rho, lap_vals=lap,
                          Phi_self=phi_s, Phi_neigh=phi_n)
    neigh_blk = _hop_block(0, 1, layer=0, rho=rho, lap_vals=lap,
                           Phi_self=phi_s, Phi_neigh=phi_n)
    assert torch.allclose(self_blk, 0.5 * phi_s)        # 0,5*2 = 1.0
    assert torch.allclose(neigh_blk, -0.3 * phi_n)      # -0.3*1 = -0.3

def test_J_hom_two_hops():
    rho, lap, phi_s, phi_n = _mini_blocks()
    path = [0, 0, 1]                                 # self-hop then neighbour
    J = _J_hom(path, rho, lap, phi_s, phi_n)               # scalar 1 by 1 tensor
    expect = (-0.4)                                 
    assert torch.allclose(J, torch.tensor([[expect]]), atol=1e-9)

def test_J_tele_scaling_and_tau():
    rho, lap, phi_s, phi_n = _mini_blocks()
    tail = [0, 1]          # one hop that lives in layer index 1
    tau = 0.2
    J = _J_tele(tail, m=0, tau=tau, rho=rho, lap_vals=lap,
                Phi_self=phi_s, Phi_neigh=phi_n)
    expect = tau * (-0.4)     # −0.3 from layer 1 neighbour hop
    assert torch.allclose(J, torch.tensor([[expect]]), atol=1e-9)

@pytest.fixture(scope="module")
def toy():
    device = "cpu"
    # graph meta
    n, L = 3, 2
    d_in = d_out = 1
    # layer-specific parameters
    rho_list = [
        torch.tensor([0.9, 1.1, 1.0], device=device),
        torch.tensor([1.0, 0.8, 1.2], device=device),
    ]

    lap_list = [
        {(1, 0): 0.5, (2, 1): 0.4},          # L^{(1)}
        {(2, 0): 0.3, (1, 2): 0.6},          # L^{(2)}
    ]

    # Φ blocks
    Phi_self  = torch.tensor([[2.0]], device=device)   # 1 by 1 for scalar features
    Phi_neigh = torch.tensor([[1.5]], device=device)

    # adjacency, including self loops, for path enumeration
    layer_graphs = [
        {0: [0, 1], 1: [1, 2], 2: [2]},               # layer 1
        {0: [0, 2], 1: [1],    2: [2, 1]},            # layer 2
    ]

    return n, L, layer_graphs, rho_list, lap_list, Phi_self, Phi_neigh

@pytest.mark.parametrize("u,v", [(0, 2), (1, 1), (2, 0)])
def test_homogeneous_matches_dense(toy, u, v):
    n, L, layer_graphs, rho, lap, phis, phin = toy
    τ = 0.0
    # reference via dense products
    layer_ops = build_layer_ops(rho, lap, phis, phin)
    J_dense   = depth_jacobian(layer_ops, tau=τ)        
    J_block   = extract_block(J_dense, v, u, phis.shape[0], phis.shape[1])
    # path enumeration answer
    J_path = pathwise_jacobian(
        u, v, L,
        layer_graphs,
        rho, lap,
        phis, phin,
        tau=τ,
    )

    assert torch.allclose(J_path, J_block, atol=1e-5)

@pytest.mark.parametrize("u,v", [(0, 2), (1, 2)])
def test_teleport_matches_dense(toy, u, v):
    n, L, layer_graphs, rho, lap, phis, phin = toy
    τ = 0.25

    layer_ops = build_layer_ops(rho, lap, phis, phin)
    J_dense   = depth_jacobian(layer_ops, tau=τ)
    J_block   = extract_block(J_dense, v, u, phis.shape[0], phis.shape[1])

    J_path = pathwise_jacobian(
        u, v, L,
        layer_graphs,
        rho, lap,
        phis, phin,
        tau=τ,
    )

    assert torch.allclose(J_path, J_block, atol=1e-5)

def test_depth_one_edge_case():
    n, L = 2, 1
    rho = [torch.tensor([0.7, 1.3])]
    Lvals = [{(1, 0): 0.4}]
    phis = torch.tensor([[2.0]])
    phin = torch.tensor([[1.0]])
    layer_graphs = [{0: [0, 1], 1: [1]}]


    τ = 0.5
    J_uv = pathwise_jacobian(
        0, 0, L, layer_graphs, rho, Lvals, phis, phin, tau=τ
    )
    expected = rho[0][0] * phis + τ * torch.eye(1)
    assert torch.allclose(J_uv, expected, atol=1e-6)

    J_10 = pathwise_jacobian(
        0, 1, L, layer_graphs, rho, Lvals, phis, phin, tau=τ
    )
    expected = -Lvals[0][(1, 0)] * phin 
    assert torch.allclose(J_10, expected, atol=1e-6)

def test_layer_jacobian_sparse_shape_and_density():
    J = _toy_layer_jacobian()
    assert J.shape == (4, 4)
    assert J._values().numel() >= 8

def test_depth_jacobian_product_and_tau():
    A = torch.tensor([[2., 0.],
                      [0., 1.]])
    B = torch.tensor([[1., 1.],
                      [0., 1.]])
    expect = B @ A
    assert torch.allclose(depth_jacobian([A, B], tau=0.0), expect)
    tau = 0.5
    expected_tau = expect + tau * (torch.eye(2) + B)
    assert torch.allclose(depth_jacobian([A, B], tau=tau),
                          expected_tau,
                          atol=1e-6)
    
def test_pathwise_amplification_mixed_hops():
    path = [0, 1, 1]
    rho = torch.tensor([0.7, 0.6])
    lap = {(1, 0): -0.8, (0, 1): -0.8}
    Phi_s = torch.eye(1)
    Phi_n = torch.tensor([[1.]])
    gamma = pathwise_amplification(path, rho, lap, Phi_s, Phi_n)
    assert math.isclose(gamma, 0.8 * 1.0 * 0.6 * 1.0, rel_tol=1e-6)

# ----------------------------------------------------------------------
#  Section 5  ──  τ-budget allocator
# ----------------------------------------------------------------------

def test_allocate_tau_budget_basic_properties():
    s_self, s_neigh, tau = allocate_tau_budget(rho_max=1.0)
    assert 0.0 < s_neigh < 0.5
    assert s_self == pytest.approx(2.0 * s_neigh, rel=1e-6)
    assert 0.0 < tau < 1.0

# ----------------------------------------------------------------------
# Main execution for testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
