import argparse

import numpy as np
import pytest
import torch

# Skip GPU-dependent tests when CUDA is unavailable.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def make_gaussian_model(n=100):
    """Build a minimal GaussianModel from n random points.

    Heavy imports are performed inside the function body so that test
    collection works even before the CUDA extensions are built.
    """
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud

    points = np.random.rand(n, 3).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

    model = GaussianModel(sh_degree=3)
    model.create_from_pcd(pcd, spatial_lr_scale=5)
    return model


def make_optimization_params():
    """Return the parsed namespace carrying OptimizationParams defaults.

    Suitable for passing to GaussianModel.training_setup.
    """
    from arguments import OptimizationParams

    parser = argparse.ArgumentParser()
    OptimizationParams(parser)
    return parser.parse_args([])
