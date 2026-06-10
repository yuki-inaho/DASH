import sys
from argparse import ArgumentParser
from arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    apply_cli_overrides,
    collect_explicit_cli_overrides,
    merge_hydra_config,
    merge_config,
)

def test_n3dv_config_overrides():
    parser = ArgumentParser()
    ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
    args = parser.parse_args(["-s", "/tmp", "--model_path", "/tmp/out"])
    args = merge_config(args, "arguments/n3dv.py")
    assert args.scale_xyz == [0.4, 0.5, 1.0]
    assert args.opacity_reset_interval == 6000
    assert args.lambda_dssim == 0.2


def test_explicit_cli_overrides_win_after_config():
    parser = ArgumentParser()
    ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000])
    argv = [
        "-s", "/tmp",
        "--model_path", "/tmp/out",
        "--conf", "arguments/n3dv.py",
        "--iterations", "1",
        "--test_iterations", "1",
    ]
    args = parser.parse_args(argv)
    cli_overrides = collect_explicit_cli_overrides(parser, args, argv)
    args = merge_config(args, args.conf)
    args = apply_cli_overrides(args, cli_overrides)

    assert args.iterations == 1
    assert args.test_iterations == [1]
    assert args.scale_xyz == [0.4, 0.5, 1.0]


def test_hydra_config_merges_with_legacy_config_and_cli_overrides(tmp_path):
    hydra_config = tmp_path / "dash_smoke.yaml"
    hydra_config.write_text(
        "\n".join(
            [
                "iterations: 8",
                "gaussian_optimizer_type: muon_schedulefree",
                "deform_optimizer_type: adamw_schedulefree",
                "scale_xyz: [0.7, 0.8, 0.9]",
            ]
        ),
        encoding="utf-8",
    )

    parser = ArgumentParser()
    ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--hydra_config", type=str, default=None)
    parser.add_argument("--hydra_overrides", nargs="*", default=[])
    argv = [
        "-s", "/tmp",
        "--model_path", "/tmp/out",
        "--conf", "arguments/n3dv.py",
        "--hydra_config", str(hydra_config),
        "--hydra_overrides", "lambda_dssim=0.33",
        "--iterations", "2",
    ]
    args = parser.parse_args(argv)
    cli_overrides = collect_explicit_cli_overrides(parser, args, argv)
    args = merge_config(args, args.conf)
    args = merge_hydra_config(args, args.hydra_config, args.hydra_overrides)
    args = apply_cli_overrides(args, cli_overrides)

    assert args.iterations == 2
    assert args.lambda_dssim == 0.33
    assert args.scale_xyz == [0.7, 0.8, 0.9]
    assert args.gaussian_optimizer_type == "muon_schedulefree"
    assert args.deform_optimizer_type == "adamw_schedulefree"
