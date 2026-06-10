from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Literal, cast

import muon
import schedulefree
import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

OptimizerType = Literal["adam", "adamw_schedulefree", "muon", "muon_schedulefree"]
ScheduleFreeOptimizer = schedulefree.ScheduleFreeWrapper | schedulefree.ScheduleFreeWrapperReference


class SingleDeviceMuonWithAuxAdam(Optimizer):
    """Single-GPU adapter around KellerJordan/Muon update functions.

    KellerJordan's distributed optimizer initializes state with ``len(state)``.
    schedulefree adds a ``z`` buffer before the inner optimizer step, so this
    adapter checks individual state keys instead.
    """

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param)
            update = muon.muon_update(
                param.grad,
                state["momentum_buffer"],
                beta=group["momentum"],
                ns_steps=group["ns_steps"],
            )
            param.mul_(1 - group["lr"] * group["weight_decay"])
            param.add_(update.reshape(param.shape), alpha=-group["lr"])

    def _step_adam_group(self, group: dict[str, Any]) -> None:
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] = 0
            state["step"] += 1
            update = muon.adam_update(
                param.grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
                group["betas"],
                group["eps"],
            )
            param.mul_(1 - group["lr"] * group["weight_decay"])
            param.add_(update, alpha=-group["lr"])


@beartype
def normalize_optimizer_type(optimizer_type: str) -> OptimizerType:
    normalized = optimizer_type.strip().lower().replace("-", "_")
    aliases = {
        "schedulefree": "adamw_schedulefree",
        "schedulerfree": "adamw_schedulefree",
        "schedule_free": "adamw_schedulefree",
        "muon_schedule_free": "muon_schedulefree",
        "muon_schedulerfree": "muon_schedulefree",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"adam", "adamw_schedulefree", "muon", "muon_schedulefree"}:
        raise ValueError(
            "Unsupported optimizer_type "
            f"{optimizer_type!r}; expected adam, adamw_schedulefree, muon, or muon_schedulefree"
        )
    return cast(OptimizerType, normalized)


@beartype
def build_optimizer(
    param_groups: Iterable[dict[str, Any]],
    optimizer_type: str,
    training_args: Any,
    *,
    eps: float = 1e-15,
) -> Optimizer | ScheduleFreeOptimizer:
    groups = [dict(group) for group in param_groups]
    normalized = normalize_optimizer_type(optimizer_type)
    if normalized == "adam":
        return torch.optim.Adam(groups, lr=0.0, eps=eps)
    if normalized == "adamw_schedulefree":
        optimizer = schedulefree.AdamWScheduleFree(
            groups,
            lr=0.0,
            weight_decay=float(getattr(training_args, "optimizer_weight_decay", 0.0)),
        )
        optimizer.train()
        return optimizer

    base = _build_single_device_muon(groups, training_args)
    if normalized == "muon":
        return base

    optimizer = schedulefree.ScheduleFreeWrapperReference(
        base,
        momentum=float(getattr(training_args, "schedulefree_momentum", 0.9)),
        weight_decay_at_y=float(getattr(training_args, "schedulefree_weight_decay_at_y", 0.0)),
    )
    optimizer.train()
    return optimizer


@beartype
def optimizer_display_name(optimizer: Optimizer | ScheduleFreeOptimizer) -> str:
    if isinstance(optimizer, (schedulefree.ScheduleFreeWrapper, schedulefree.ScheduleFreeWrapperReference)):
        return f"schedulefree({optimizer.base.__class__.__name__})"
    return optimizer.__class__.__name__


@beartype
def set_optimizer_mode(optimizer: Any, mode: Literal["train", "eval"]) -> None:
    mode_fn = getattr(optimizer, mode, None)
    if callable(mode_fn):
        mode_fn()


@jaxtyped(typechecker=beartype)
def replace_optimizer_parameter(
    optimizer: Optimizer | ScheduleFreeOptimizer,
    group: MutableMapping[str, Any],
    tensor: Float[Tensor, "..."],
) -> Parameter:
    old_param = group["params"][0]
    state = optimizer.state.pop(old_param, None)
    new_param = Parameter(tensor.requires_grad_(True))
    group["params"][0] = new_param
    if state is not None:
        optimizer.state[new_param] = _reset_state_for_replacement(state, tensor)
    return new_param


@jaxtyped(typechecker=beartype)
def prune_optimizer_parameter(
    optimizer: Optimizer | ScheduleFreeOptimizer,
    group: MutableMapping[str, Any],
    mask: Bool[Tensor, "points"],
) -> Parameter:
    old_param = group["params"][0]
    state = optimizer.state.pop(old_param, None)
    new_tensor = old_param[mask].requires_grad_(True)
    new_param = Parameter(new_tensor)
    group["params"][0] = new_param
    if state is not None:
        optimizer.state[new_param] = _slice_state_on_first_dim(state, mask)
    return new_param


@jaxtyped(typechecker=beartype)
def extend_optimizer_parameter(
    optimizer: Optimizer | ScheduleFreeOptimizer,
    group: MutableMapping[str, Any],
    extension_tensor: Float[Tensor, "..."],
) -> Parameter:
    old_param = group["params"][0]
    state = optimizer.state.pop(old_param, None)
    new_tensor = torch.cat((old_param, extension_tensor), dim=0).requires_grad_(True)
    new_param = Parameter(new_tensor)
    group["params"][0] = new_param
    if state is not None:
        optimizer.state[new_param] = _extend_state_on_first_dim(state, extension_tensor)
    return new_param


@beartype
def _build_single_device_muon(
    param_groups: list[dict[str, Any]],
    training_args: Any,
) -> Optimizer:
    converted_groups: list[dict[str, Any]] = []
    group_names: list[str | None] = []
    for group in param_groups:
        name = group.get("name")
        muon_params: list[Tensor] = []
        adam_params: list[Tensor] = []
        for param in group["params"]:
            if param.ndim in (2, 4):
                muon_params.append(param)
            else:
                adam_params.append(param)

        weight_decay = float(getattr(training_args, "optimizer_weight_decay", 0.0))
        if muon_params:
            converted_groups.append(
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": float(group.get("lr", 0.02)),
                    "momentum": float(getattr(training_args, "muon_momentum", 0.95)),
                    "ns_steps": int(getattr(training_args, "muon_ns_steps", 5)),
                    "weight_decay": weight_decay,
                }
            )
            group_names.append(name)
        if adam_params:
            converted_groups.append(
                {
                    "params": adam_params,
                    "use_muon": False,
                    "lr": float(group.get("lr", 3e-4)),
                    "betas": (0.9, 0.95),
                    "eps": 1e-10,
                    "weight_decay": weight_decay,
                }
            )
            group_names.append(name)

    if not converted_groups:
        raise ValueError("Cannot build optimizer without trainable parameters")

    optimizer = SingleDeviceMuonWithAuxAdam(converted_groups)
    for group, name in zip(optimizer.param_groups, group_names):
        if name is not None:
            group["name"] = name
    return optimizer


@beartype
def _reset_state_for_replacement(state: MutableMapping[str, Any], tensor: Tensor) -> MutableMapping[str, Any]:
    for key, value in list(state.items()):
        if torch.is_tensor(value) and value.shape == tensor.shape:
            state[key] = tensor.detach().clone() if key == "z" else torch.zeros_like(tensor)
    return state


@jaxtyped(typechecker=beartype)
def _slice_state_on_first_dim(
    state: MutableMapping[str, Any],
    mask: Bool[Tensor, "points"],
) -> MutableMapping[str, Any]:
    for key, value in list(state.items()):
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == mask.shape[0]:
            state[key] = value[mask]
    return state


@jaxtyped(typechecker=beartype)
def _extend_state_on_first_dim(
    state: MutableMapping[str, Any],
    extension_tensor: Float[Tensor, "..."],
) -> MutableMapping[str, Any]:
    for key, value in list(state.items()):
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[1:] == extension_tensor.shape[1:]
        ):
            extension = extension_tensor.detach().clone() if key == "z" else torch.zeros_like(extension_tensor)
            state[key] = torch.cat((value, extension), dim=0)
    return state
