import sys
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, merge_config

def test_n3dv_config_overrides():
    parser = ArgumentParser()
    ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
    args = parser.parse_args(["-s", "/tmp", "--model_path", "/tmp/out"])
    args = merge_config(args, "arguments/n3dv.py")
    assert args.scale_xyz == [0.4, 0.5, 1.0]
    assert args.opacity_reset_interval == 6000
    assert args.lambda_dssim == 0.2
