from argparse import Namespace

import muon
import schedulefree
import torch
import pytest

from utils.optimizer_utils import (
    build_optimizer,
    extend_optimizer_parameter,
    normalize_optimizer_type,
    optimizer_display_name,
    prune_optimizer_parameter,
    replace_optimizer_parameter,
    set_optimizer_mode,
    SingleDeviceMuonWithAuxAdam,
)


def _args():
    return Namespace(
        optimizer_weight_decay=0.0,
        muon_momentum=0.95,
        muon_ns_steps=5,
        schedulefree_momentum=0.9,
        schedulefree_weight_decay_at_y=0.0,
    )


def test_optimizer_type_aliases_and_rejects_unknown():
    assert normalize_optimizer_type("schedulerfree") == "adamw_schedulefree"
    assert normalize_optimizer_type("muon-schedule-free") == "muon_schedulefree"
    with pytest.raises(ValueError, match="Unsupported optimizer_type"):
        normalize_optimizer_type("dummy")


def test_adam_factory_preserves_group_name():
    param = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = build_optimizer([{"params": [param], "lr": 0.1, "name": "xyz"}], "adam", _args())

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["name"] == "xyz"


def test_muon_factory_splits_matrix_and_aux_parameters():
    matrix = torch.nn.Parameter(torch.ones(2, 3))
    bias = torch.nn.Parameter(torch.ones(3))
    optimizer = build_optimizer(
        [{"params": [matrix, bias], "lr": 0.1, "name": "deform"}],
        "muon",
        _args(),
    )

    assert hasattr(muon, "muon_update")
    assert isinstance(optimizer, SingleDeviceMuonWithAuxAdam)
    assert {group["use_muon"] for group in optimizer.param_groups} == {True, False}
    assert all(group["name"] == "deform" for group in optimizer.param_groups)

    loss = matrix.square().sum() + bias.square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert not torch.equal(matrix.detach(), torch.ones_like(matrix))
    assert not torch.equal(bias.detach(), torch.ones_like(bias))


def test_muon_schedulefree_factory_steps_in_train_mode():
    param = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = build_optimizer(
        [{"params": [param], "lr": 0.1, "name": "grid"}],
        "muon_schedulefree",
        _args(),
    )

    assert isinstance(optimizer, schedulefree.ScheduleFreeWrapperReference)
    assert "SingleDeviceMuonWithAuxAdam" in optimizer_display_name(optimizer)

    param.square().sum().backward()
    set_optimizer_mode(optimizer, "train")
    optimizer.step()
    optimizer.zero_grad()
    set_optimizer_mode(optimizer, "eval")

    assert not torch.equal(param.detach(), torch.ones_like(param))


def test_optimizer_state_helpers_are_not_adam_only():
    param = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    group = {"params": [param], "lr": 0.1, "name": "xyz"}
    optimizer = build_optimizer([group], "adam", _args())
    param.sum().backward()
    optimizer.step()

    extension = torch.ones(1, 3)
    extended = extend_optimizer_parameter(optimizer, optimizer.param_groups[0], extension)
    assert extended.shape == (3, 3)
    assert optimizer.state[extended]["exp_avg"].shape == (3, 3)

    pruned = prune_optimizer_parameter(
        optimizer,
        optimizer.param_groups[0],
        torch.tensor([True, False, True]),
    )
    assert pruned.shape == (2, 3)
    assert optimizer.state[pruned]["exp_avg"].shape == (2, 3)

    replacement = replace_optimizer_parameter(
        optimizer,
        optimizer.param_groups[0],
        torch.zeros(2, 3),
    )
    assert replacement.shape == (2, 3)
    assert torch.equal(optimizer.state[replacement]["exp_avg"], torch.zeros(2, 3))
