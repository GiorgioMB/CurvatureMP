import torch
import pytest
from ..CGMP.layer import CurvatureGatedMessagePropagationLayer, _is_undirected

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def toy_triangle_graph() -> torch.LongTensor:
    return torch.tensor([
        [0, 0, 1, 1, 2, 2],  # sources
        [1, 2, 0, 2, 0, 1],  # targets
    ], dtype=torch.long)


def random_node_features(seed: int = 0, *, num_nodes: int = 3, dim: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(num_nodes, dim)


def spectral_norm(matrix: torch.Tensor) -> float:
    return torch.linalg.svdvals(matrix)[0].item()



# ---------------------------------------------------------------------------
#  unit tests
# ---------------------------------------------------------------------------
def test_is_undirected_true():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    assert _is_undirected(edge_index, num_nodes=3) is True


def test_is_undirected_false():
    edge_index = torch.tensor([[0, 1],
                               [1, 2]], dtype=torch.long)
    assert _is_undirected(edge_index, num_nodes=3) is False


def test_is_undirected_self_loops_only():
    edge_index = torch.tensor([[0, 1],
                               [0, 1]], dtype=torch.long) 
    assert _is_undirected(edge_index, num_nodes=2) is True


def test_forward_shape_and_types():
    layer = CurvatureGatedMessagePropagationLayer(
        4, 4, tau=0.30, spectral_caps=(0.20, 0.10)
    )
    x = random_node_features()
    edge_index = toy_triangle_graph()

    h_next, edge_index_new, w_new = layer(x, edge_index)
    assert h_next.shape == x.shape
    assert h_next.dtype == x.dtype
    assert edge_index_new.shape[0] == 2
    assert w_new.numel() == edge_index_new.size(1)


def test_tau_is_adapted_when_unspecified():
    layer = CurvatureGatedMessagePropagationLayer(4, 4, tau=None)
    x = random_node_features()
    edge_index = toy_triangle_graph()
    _ = layer(x, edge_index)  # forward once
    tau_val = layer.tau.item()
    assert 0.0 < tau_val < 1.0, "tau should be set inside (0,1) after first pass"


def test_spectral_caps_enforced():
    caps = (0.05, 0.05)
    layer = CurvatureGatedMessagePropagationLayer(4, 4, tau=0.25, spectral_caps=caps)
    layer.enforce_spectral_caps()
    sigma_self = spectral_norm(layer.phi_self.weight)
    sigma_neigh = spectral_norm(layer.phi_neigh.weight)
    assert sigma_self <= caps[0] + 1e-4
    assert sigma_neigh <= caps[1] + 1e-4


@pytest.mark.parametrize("tau", [0.15, 0.50])
def test_backward_pass(tau):
    layer = CurvatureGatedMessagePropagationLayer(4, 4, tau=tau)
    x = random_node_features().requires_grad_()
    edge_index = toy_triangle_graph()
    out, _, _ = layer(x, edge_index)
    loss = out.pow(2).mean()
    loss.backward()
    # gradients flow to both phi_self and phi_neigh
    assert layer.phi_self.weight.grad is not None
    assert layer.phi_neigh.weight.grad is not None
    assert torch.isfinite(layer.phi_self.weight.grad).all()
    assert torch.isfinite(layer.phi_neigh.weight.grad).all()


def test_edge_weights_remain_positive():
    layer = CurvatureGatedMessagePropagationLayer(4, 4, tau=0.30)
    x = random_node_features()
    edge_index = toy_triangle_graph()
    _, _, w_new = layer(x, edge_index)
    assert torch.all(w_new > 0), "Ricci-flow step must keep weights positive"


@pytest.mark.parametrize("in_ch,out_ch", [(4, 10), (10, 4)])
def test_adapter_is_isometry(in_ch, out_ch):
    layer = CurvatureGatedMessagePropagationLayer(in_ch, out_ch)
    if not layer._needs_adapter:
        pytest.skip("no adapter when dimensions match")
    P = layer.P        # (m, n) plain tensor
    m, n = P.shape

    if m >= n:                           # tall  (m ≥ n)
        eye = torch.eye(n, device=P.device)
        assert torch.allclose(P.T @ P, eye, atol=1e-5, rtol=1e-5)
    else:                                # wide  (m < n)
        eye = torch.eye(m, device=P.device)
        assert torch.allclose(P @ P.T, eye, atol=1e-5, rtol=1e-5)
    # singular values ≈ 1
    s = torch.linalg.svdvals(P)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5, rtol=1e-5)


def test_norm_preservation_tall():
    layer = CurvatureGatedMessagePropagationLayer(6, 12)  
    x = torch.randn(128, 6)
    y = layer.h0_adapter(x)                               
    assert torch.allclose(x.norm(dim=1), y.norm(dim=1),
                          atol=1e-5, rtol=1e-5)


def test_adapter_is_frozen():
    layer = CurvatureGatedMessagePropagationLayer(5, 9)
    # a) not in the trainable parameter list
    trainables = {n for n, p in layer.named_parameters() if p.requires_grad}
    assert "P" not in trainables
    # b) buffer flag is True
    assert "P" in dict(layer.named_buffers())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
