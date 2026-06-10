import torch
from tests.conftest import make_gaussian_model, requires_cuda

@requires_cuda
def test_dynamic_is_bool_after_create():
    g = make_gaussian_model(100)
    assert g.get_dynamic.dtype == torch.bool

@requires_cuda
def test_inverted_mask_selects_all():
    g = make_gaussian_model(100)
    mask = g.get_dynamic.squeeze(1)
    # bool: ~(all False) = all True -> identity selection. long: ~0 = -1 -> wrong rows.
    assert torch.equal(g.get_xyz[~mask], g.get_xyz)

@requires_cuda
def test_set_dynamic_keeps_bool():
    g = make_gaussian_model(100)
    n = g.get_xyz.shape[0]
    g.set_dynamic(torch.ones(n, 1))
    assert g.get_dynamic.dtype == torch.bool
