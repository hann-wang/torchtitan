# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Relative update distance of Linear and grouped-GEMM weights.

Vanishing updates show up as a stored weight that barely moves relative to
its own magnitude. For each ``nn.Linear`` weight and each ``GroupedLinear``
(grouped GEMM) weight, the logged value is

    ||W_after - W_before||_2 / ||W_before||_2

``W_before`` is the stored weight at the start of the optimizer step and
``W_after`` is the stored weight when that step returns, so optimizer
post-hooks are included on the step that applies them.

Distances are recorded on the same steps as the other metrics (``log_freq``,
and always step 1). A logged step reports that step's update, not the sum of
updates since the previous log. Set ``metrics.log_freq`` to 1 to see every
optimizer step.

Sharded weights use the global L2 norm (``DTensor.full_tensor`` on the norm).
Pipeline stages hold different weights, so the per-rank results are gathered
before logging. Only the metrics rank's logger writes them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.linear import GroupedLinear

logger = logging.getLogger(__name__)

_LINEAR_PREFIX = "weight_update/linear/"
_GMM_PREFIX = "weight_update/gmm/"


def relative_l2_distance(before: torch.Tensor, after: torch.Tensor) -> float:
    """``||after - before||_2 / ||before||_2``, or 0 when ``before`` is 0.

    Norms are computed in fp32. A DTensor norm is reduced across its mesh,
    which is ``sqrt(sum(local_squares))`` rather than the sum of local norms.
    """
    delta_norm = _full_l2(after.detach().float() - before.detach().float())
    base_norm = _full_l2(before.detach().float())
    base_value = float(base_norm)
    if base_value == 0.0:
        return 0.0
    return float(delta_norm) / base_value


class WeightUpdateMonitor:
    """Snapshots linear and grouped-GEMM weights around an optimizer step."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._tracker: _WeightUpdateTracker | None = None
        self._pending: dict[str, float] | None = None

    @contextmanager
    def observe(
        self,
        model_parts: list[nn.Module] | None,
        active: bool,
    ) -> Iterator[None]:
        """Snapshot on entry and reduce distances on a clean exit.

        ``active`` must match ``MetricsProcessor.should_log`` for this step,
        and every rank must pass the same value. The distance reduction and
        the pipeline gather are collectives.
        """
        if not self.enabled or not active or model_parts is None:
            yield
            return
        tracker = self._ensure_tracker(model_parts)
        tracker.begin()
        try:
            yield
        except BaseException:
            tracker.abort()
            self._pending = None
            raise
        self._pending = tracker.finish()

    def consume(self) -> dict[str, float]:
        pending = self._pending or {}
        self._pending = None
        return pending

    def _ensure_tracker(self, model_parts: list[nn.Module]) -> _WeightUpdateTracker:
        if self._tracker is None:
            specs = list(_iter_tracked_weights(model_parts))
            self._tracker = _WeightUpdateTracker(specs)
            logger.info(
                "Relative weight-update tracking enabled "
                "(%d linear/gmm weights on this rank)",
                len(specs),
            )
        return self._tracker


def _weight_kind(module: nn.Module) -> str | None:
    # GroupedLinear is not an nn.Linear. Check it first so a future subclass
    # of both is still reported as a grouped GEMM.
    if isinstance(module, GroupedLinear):
        return "gmm"
    if isinstance(module, nn.Linear):
        return "linear"
    return None


def _iter_tracked_weights(
    model_parts: list[nn.Module],
) -> Iterator[tuple[str, nn.Parameter]]:
    seen: set[int] = set()
    for part in model_parts:
        for module_fqn, module in part.named_modules():
            kind = _weight_kind(module)
            if kind is None:
                continue
            weight = getattr(module, "weight", None)
            if (
                not isinstance(weight, nn.Parameter)
                or not weight.requires_grad
                or not weight.is_floating_point()
                or weight.numel() == 0
            ):
                continue
            if id(weight) in seen:
                continue
            seen.add(id(weight))
            name = module_fqn if module_fqn else type(module).__name__
            prefix = _GMM_PREFIX if kind == "gmm" else _LINEAR_PREFIX
            yield f"{prefix}{name}", weight


def _full_l2(tensor: torch.Tensor) -> torch.Tensor:
    norm = tensor.norm()
    if isinstance(norm, DTensor):
        return norm.full_tensor()
    return norm


def _summarize(distances: list[float]) -> dict[str, float]:
    if not distances:
        return {}
    ordered = sorted(distances)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "weight_update/min_relative_distance": ordered[0],
        "weight_update/median_relative_distance": median,
        "weight_update/max_relative_distance": ordered[-1],
    }


def _merge_across_ranks(local: dict[str, float]) -> dict[str, float]:
    """Union per-rank distances.

    Data-parallel replicas publish the same global ratio. Pipeline ranks
    publish different weights. ``setdefault`` keeps one copy of a key.
    """
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() < 2
    ):
        merged = dict(local)
    else:
        gathered: list[dict[str, float] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        merged = {}
        for part in gathered:
            if not part:
                continue
            for key, value in part.items():
                merged.setdefault(key, value)
    distances = [
        value
        for key, value in merged.items()
        if key.startswith(_LINEAR_PREFIX) or key.startswith(_GMM_PREFIX)
    ]
    merged.update(_summarize(distances))
    return merged


class _WeightUpdateTracker:
    def __init__(self, specs: list[tuple[str, nn.Parameter]]) -> None:
        self._specs = specs
        self._snapshots: dict[int, torch.Tensor] | None = None

    @torch.no_grad()
    def begin(self) -> None:
        self._snapshots = {
            id(param): param.detach().clone() for _, param in self._specs
        }

    def abort(self) -> None:
        self._snapshots = None

    @torch.no_grad()
    def finish(self) -> dict[str, float]:
        snapshots = self._snapshots
        self._snapshots = None
        if snapshots is None:
            return {}
        local: dict[str, float] = {}
        for tag, param in self._specs:
            before = snapshots.pop(id(param), None)
            if before is None:
                continue
            local[tag] = relative_l2_distance(before, param)
        return _merge_across_ranks(local)
