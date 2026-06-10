import torch
from tests.conftest import make_gaussian_model, make_optimization_params, requires_cuda

@requires_cuda
def test_low_opacity_pruned_without_screen_size():
    from utils.general_utils import inverse_sigmoid
    g = make_gaussian_model(100)
    g.training_setup(make_optimization_params())
    n0 = g.get_xyz.shape[0]
    with torch.no_grad():
        g._opacity[:50] = inverse_sigmoid(torch.tensor(0.001, device="cuda"))
    g.densify_and_prune(max_grad=1e9, min_opacity=0.005, extent=1.0, max_screen_size=None)
    assert g.get_xyz.shape[0] < n0

@requires_cuda
def test_big_point_prune_with_screen_size():
    g = make_gaussian_model(100)
    g.training_setup(make_optimization_params())
    n0 = g.get_xyz.shape[0]
    with torch.no_grad():
        g.max_radii2D[:30] = 1000.0
    g.densify_and_prune(max_grad=1e9, min_opacity=0.0, extent=1.0, max_screen_size=20)
    assert g.get_xyz.shape[0] < n0
