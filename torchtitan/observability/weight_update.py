# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Relative update distance of Linear and grouped-GEMM weights, plus the
weight-side failure modes of low-precision training.

For each ``nn.Linear`` weight and each ``GroupedLinear`` (grouped GEMM)
weight, the logged distance is

    ||W_after - W_before||_2 / ||W_before||_2

``W_before`` is the stored weight at the start of the optimizer step and
``W_after`` is the stored weight when that step returns, so optimizer
post-hooks are included on the step that applies them.

The same pair of snapshots produces ``fp4_failure/`` scalars. They are
summaries across every tracked weight, not a second copy of the per-layer
distance:

- A, update vanishing. ``A_sub_ulp_fraction`` is the share of elements whose
  stored step is smaller than half an ulp of the stored value, so the add
  rounds away. ``A_median_update_over_ulp`` and ``A_min_update_over_ulp`` are
  the median and worst-layer median of ``|Δw| / ulp(w)`` over elements that
  actually moved. On an FP32 master the fraction stays near 0 and the ratio
  stays large. On BF16 or FP4 the fraction rises and the ratio falls toward 1
  before updates disappear.
- B, multiplicative updates near zero and across signs. ``B_exact_zero_fraction``
  and ``B_near_zero_fraction`` (elements smaller than 1% of that tensor's RMS)
  are coordinates where a relative step barely moves. ``B_sign_flip_fraction``
  is coordinates whose sign changed, which a multiplicative update cannot do.
- C, common-mode drift of the output projection (the LM head, or Flux
  ``final_layer.linear``). ``C_head_row_mean_rms`` is the RMS of the per-row
  means, ``C_head_row_mean_relative_drift`` is how far that vector moved this
  step, and ``C_head_update_mean_over_rms`` is ``|mean(ΔW)| / RMS(ΔW)``. A
  large last value means the update is not zero-sum across output rows.
  Logit quantization error and predictive KL need a second-precision forward
  and are not logged here.
- E, outliers and oscillation. ``E_outlier_fraction`` is the share of elements
  larger than 6 times the tensor RMS, ``E_max_over_rms`` is the worst peak,
  and ``E_update_sign_reversal_fraction`` is the share of coordinates whose
  nonzero step changed sign since the previous logged step.

Attention score conservation (failure mode D) is not logged. The scores and
``dS`` stay inside the flash / varlen kernel, so a weight snapshot cannot see
``1ᵀ dS``.

Distances and failure stats are recorded on the same steps as the other
metrics (``log_freq``, and always step 1). A logged step reports that step's
update, not the sum of updates since the previous log. Set
``metrics.log_freq`` to 1 to see every optimizer step.

Sharded weights use the global L2 norm (``DTensor.full_tensor`` on the norm).
Element fractions are computed on the local shard. Pipeline stages hold
different weights, so the per-rank results are gathered before logging. Only
the metrics rank's logger writes them.
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
# A multiplicative step on a coordinate this small, relative to the tensor RMS,
# is a negligible absolute move.
_NEAR_ZERO_RMS_RATIO = 1e-2
_OUTLIER_SIGMA = 6.0


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


def _as_local(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _storage_ulp(weight: torch.Tensor) -> torch.Tensor:
    """Gap from each stored element to the neighbor farther from zero.

    This is the local spacing of the weight's dtype. An additive step smaller
    than half of it rounds back to the same stored value.
    """
    positive = weight >= 0
    toward_pos = torch.nextafter(weight, torch.full_like(weight, float("inf")))
    toward_neg = torch.nextafter(weight, torch.full_like(weight, float("-inf")))
    gap = torch.where(positive, toward_pos - weight, weight - toward_neg)
    return gap.float().abs()


def _is_head(tag: str) -> bool:
    """Output projection whose row-mean drift is a softmax no-op in exact arithmetic."""
    fqn = tag[len(_GMM_PREFIX):] if tag.startswith(_GMM_PREFIX) else tag[len(_LINEAR_PREFIX):]
    return (
        fqn == "final_layer"
        or fqn.endswith(".final_layer")
        or fqn == "final_layer.linear"
        or fqn.endswith(".final_layer.linear")
        or "lm_head" in fqn
        or "embed_out" in fqn
    )


def _failure_record(
    before: torch.Tensor,
    after: torch.Tensor,
    prev_sign: torch.Tensor | None,
) -> tuple[dict[str, float], torch.Tensor]:
    """Counts for one weight, plus the sign of this step's stored update."""
    stored_before = _as_local(before).detach()
    stored_after = _as_local(after).detach()
    before_f = stored_before.float()
    after_f = stored_after.float()
    delta = after_f - before_f
    abs_delta = delta.abs()
    ulp = _storage_ulp(stored_before)
    moved = abs_delta > 0
    n_moved = int(moved.sum())
    if n_moved:
        median_ratio = float((abs_delta[moved] / ulp[moved].clamp_min(1e-30)).median())
    else:
        median_ratio = 0.0

    rms = after_f.pow(2).mean().sqrt()
    rms_value = float(rms)
    if rms_value == 0.0:
        near_zero = int(after_f.numel())
        outlier = 0
        max_over_rms = 0.0
    else:
        near_zero = int((after_f.abs() < (_NEAR_ZERO_RMS_RATIO * rms)).sum())
        outlier = int((after_f.abs() > (_OUTLIER_SIGMA * rms)).sum())
        max_over_rms = float(after_f.abs().max() / rms)

    sign_flip = ((before_f > 0) & (after_f < 0)) | ((before_f < 0) & (after_f > 0))
    new_sign = delta.sign().to(dtype=torch.int8)
    has_prev = 0.0
    osc_both = 0.0
    osc_disagree = 0.0
    if prev_sign is not None and prev_sign.shape == new_sign.shape:
        has_prev = 1.0
        both = (new_sign != 0) & (prev_sign != 0)
        osc_both = float(both.sum())
        osc_disagree = float((both & (new_sign != prev_sign)).sum())

    if before_f.ndim >= 2:
        mean_before = before_f.mean(dim=-1)
        mean_after = after_f.mean(dim=-1)
        mean_before_norm = float(mean_before.norm())
        row_mean_rms = float(mean_after.pow(2).mean().sqrt())
        mean_delta_norm = float((mean_after - mean_before).norm())
        if mean_before_norm == 0.0:
            row_drift = 0.0 if mean_delta_norm == 0.0 else 1.0
        else:
            row_drift = mean_delta_norm / mean_before_norm
        delta_rms = float(delta.pow(2).mean().sqrt())
        update_mean_over_rms = 0.0 if delta_rms == 0.0 else float(delta.mean().abs() / delta_rms)
    else:
        row_mean_rms = 0.0
        row_drift = 0.0
        update_mean_over_rms = 0.0

    record = {
        "numel": float(before_f.numel()),
        "sub_ulp": float((abs_delta < (0.5 * ulp)).sum()),
        "n_moved": float(n_moved),
        "median_update_over_ulp": median_ratio,
        "exact_zero": float((after_f == 0).sum()),
        "near_zero": float(near_zero),
        "sign_flip": float(sign_flip.sum()),
        "outlier": float(outlier),
        "max_over_rms": max_over_rms,
        "has_prev": has_prev,
        "osc_both": osc_both,
        "osc_disagree": osc_disagree,
        "row_mean_rms": row_mean_rms,
        "row_mean_rel_drift": row_drift,
        "update_mean_over_rms": update_mean_over_rms,
    }
    return record, new_sign


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _summarize_failures(records: dict[str, dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    numel = sum(record["numel"] for record in records.values())
    if numel == 0.0:
        return {}

    def fraction(key: str) -> float:
        return sum(record[key] for record in records.values()) / numel

    summary = {
        "fp4_failure/A_sub_ulp_fraction": fraction("sub_ulp"),
        "fp4_failure/A_max_sub_ulp_fraction": max(
            record["sub_ulp"] / record["numel"] for record in records.values()
        ),
        "fp4_failure/B_exact_zero_fraction": fraction("exact_zero"),
        "fp4_failure/B_near_zero_fraction": fraction("near_zero"),
        "fp4_failure/B_sign_flip_fraction": fraction("sign_flip"),
        "fp4_failure/E_outlier_fraction": fraction("outlier"),
        "fp4_failure/E_max_over_rms": max(
            record["max_over_rms"] for record in records.values()
        ),
    }
    medians = [
        record["median_update_over_ulp"]
        for record in records.values()
        if record["n_moved"] > 0.0
    ]
    if medians:
        summary["fp4_failure/A_median_update_over_ulp"] = _median(medians)
        summary["fp4_failure/A_min_update_over_ulp"] = min(medians)
    heads = {tag: record for tag, record in records.items() if _is_head(tag)}
    if heads:
        summary["fp4_failure/C_head_row_mean_rms"] = max(
            record["row_mean_rms"] for record in heads.values()
        )
        summary["fp4_failure/C_head_row_mean_relative_drift"] = max(
            record["row_mean_rel_drift"] for record in heads.values()
        )
        summary["fp4_failure/C_head_update_mean_over_rms"] = max(
            record["update_mean_over_rms"] for record in heads.values()
        )
    reversible = [
        record
        for record in records.values()
        if record["has_prev"] > 0.0 and record["osc_both"] > 0.0
    ]
    if reversible:
        both = sum(record["osc_both"] for record in reversible)
        disagree = sum(record["osc_disagree"] for record in reversible)
        summary["fp4_failure/E_update_sign_reversal_fraction"] = disagree / both
    return summary


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


def _merge_across_ranks(
    local: dict[str, float],
    failures: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Union per-rank distances and failure counts.

    Data-parallel replicas publish the same global ratio. Pipeline ranks
    publish different weights. ``setdefault`` keeps one copy of a key.
    """
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() < 2
    ):
        merged = dict(local)
        merged_failures = dict(failures)
    else:
        gathered: list[dict[str, float] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        merged = {}
        for part in gathered:
            if not part:
                continue
            for key, value in part.items():
                merged.setdefault(key, value)
        gathered_failures: list[dict[str, dict[str, float]] | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(gathered_failures, failures)
        merged_failures = {}
        for part in gathered_failures:
            if not part:
                continue
            for key, value in part.items():
                merged_failures.setdefault(key, value)
    distances = [
        value
        for key, value in merged.items()
        if key.startswith(_LINEAR_PREFIX) or key.startswith(_GMM_PREFIX)
    ]
    merged.update(_summarize(distances))
    merged.update(_summarize_failures(merged_failures))
    return merged


class _WeightUpdateTracker:
    def __init__(self, specs: list[tuple[str, nn.Parameter]]) -> None:
        self._specs = specs
        self._snapshots: dict[int, torch.Tensor] | None = None
        self._prev_sign: dict[int, torch.Tensor] = {}

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
        failures: dict[str, dict[str, float]] = {}
        for tag, param in self._specs:
            before = snapshots.pop(id(param), None)
            if before is None:
                continue
            local[tag] = relative_l2_distance(before, param)
            record, new_sign = _failure_record(
                before, param, self._prev_sign.get(id(param))
            )
            failures[tag] = record
            self._prev_sign[id(param)] = new_sign
        return _merge_across_ranks(local, failures)
