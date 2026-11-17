# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.observability.logging import NoColor
from torchtitan.observability.metrics import DeviceMemStats, MetricsProcessor
from torchtitan.observability.weight_update import (
    WeightUpdateMonitor,
    _failure_record,
    _is_head,
    _storage_ulp,
    _summarize_failures,
    relative_l2_distance,
)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = Linear(Linear.Config(in_features=3, out_features=4, bias=True))
        self.plain = nn.Linear(3, 4, bias=False)
        self.experts = GroupedLinear(
            GroupedLinear.Config(group_size=2, in_features=3, out_features=4)
        )
        self.embed = nn.Embedding(5, 3)
        self.frozen = Linear(Linear.Config(in_features=3, out_features=2))
        self.tied = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.proj.weight.normal_()
            self.proj.bias.normal_()
            self.plain.weight.normal_()
            self.experts.weight.normal_()
            self.embed.weight.normal_()
            self.frozen.weight.normal_()
        self.frozen.weight.requires_grad_(False)
        self.tied.weight = self.plain.weight


class _FakeMemoryMonitor:
    def get_peak_stats(self) -> DeviceMemStats:
        return DeviceMemStats(0, 0.0, 0, 0.0, 0, 0)


def _unique_trainable(module: nn.Module) -> list[nn.Parameter]:
    seen: set[int] = set()
    params = []
    for param in module.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        params.append(param)
    return params


def _fill_grads(module: nn.Module, value: float) -> None:
    for param in _unique_trainable(module):
        param.grad = torch.full_like(param, value)


def test_relative_l2_distance_matches_vector_norm():
    before = torch.tensor([3.0, 4.0])
    after = torch.tensor([6.0, 8.0])
    assert relative_l2_distance(before, after) == pytest.approx(1.0)


def test_relative_l2_distance_zero_weight_is_zero():
    assert relative_l2_distance(torch.zeros(2, 2), torch.ones(2, 2)) == 0.0


def test_log_weight_update_distance_is_opt_in():
    assert MetricsProcessor.Config().log_weight_update_distance is False
    assert (
        MetricsProcessor.Config(enable_tensorboard=True).log_weight_update_distance
        is False
    )
    assert (
        MetricsProcessor.Config(
            enable_tensorboard=True,
            log_weight_update_distance=True,
        ).log_weight_update_distance
        is True
    )


def test_explicit_enable_without_backend_is_skipped(caplog):
    with caplog.at_level(logging.WARNING, logger="torchtitan.observability.metrics"):
        config = MetricsProcessor.Config(log_weight_update_distance=True)
    assert config.log_weight_update_distance is False
    assert "will not be recorded" in caplog.text


def test_logged_step_records_each_linear_and_gmm_weight():
    torch.manual_seed(0)
    model = _Toy()
    optim = torch.optim.SGD(_unique_trainable(model), lr=0.25)
    monitor = WeightUpdateMonitor(enabled=True)

    _fill_grads(model, 0.5)
    before = {
        "weight_update/linear/proj": model.proj.weight.detach().clone(),
        "weight_update/linear/plain": model.plain.weight.detach().clone(),
        "weight_update/gmm/experts": model.experts.weight.detach().clone(),
    }
    with monitor.observe([model], active=True):
        optim.step()
    extra = monitor.consume()

    for key, old in before.items():
        module_name = key.rsplit("/", 1)[-1]
        expected = relative_l2_distance(old, model.get_submodule(module_name).weight)
        assert extra[key] == pytest.approx(expected)
        assert extra[key] > 0.0
    assert "weight_update/linear/tied" not in extra
    assert "weight_update/linear/frozen" not in extra
    assert "weight_update/linear/embed" not in extra
    tracked = [
        value
        for key, value in extra.items()
        if key.startswith("weight_update/linear/") or key.startswith("weight_update/gmm/")
    ]
    assert extra["weight_update/min_relative_distance"] == pytest.approx(min(tracked))
    assert extra["weight_update/max_relative_distance"] == pytest.approx(max(tracked))
    assert monitor.consume() == {}

    # An inactive step still updates the weight. The next active step measures
    # only that step's move.
    _fill_grads(model, 0.5)
    with monitor.observe([model], active=False):
        optim.step()
    assert monitor.consume() == {}

    _fill_grads(model, -1.25)
    baseline = model.proj.weight.detach().clone()
    with monitor.observe([model], active=True):
        optim.step()
    extra = monitor.consume()
    assert extra["weight_update/linear/proj"] == pytest.approx(
        relative_l2_distance(baseline, model.proj.weight)
    )


def test_disabled_monitor_skips_the_snapshot():
    model = nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        model.weight.normal_()
    model.weight.grad = torch.ones_like(model.weight)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    monitor = WeightUpdateMonitor(enabled=False)
    with monitor.observe([model], active=True):
        optim.step()
    assert monitor.consume() == {}


def test_exception_discards_the_open_snapshot():
    model = nn.Linear(3, 4, bias=False)
    monitor = WeightUpdateMonitor(enabled=True)
    with pytest.raises(RuntimeError, match="boom"):
        with monitor.observe([model], active=True):
            raise RuntimeError("boom")
    assert monitor.consume() == {}


def test_training_log_includes_observed_weight_updates():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    model.weight.grad = torch.ones_like(model.weight)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    logged: dict[str, float] = {}
    processor = object.__new__(MetricsProcessor)
    processor.parallelism_context = SimpleNamespace(non_data_parallel_size=1)
    processor.color = NoColor()
    processor.logger = SimpleNamespace(log=lambda metrics, step: logged.update(metrics))
    processor.device_memory_monitor = _FakeMemoryMonitor()
    processor.ntokens_since_last_log = 10
    processor.data_loading_times = [0.1]
    processor.time_last_log = 0.0
    processor.step_last_log = 0
    processor.num_flops_per_token = 1
    processor.gpu_peak_flops = 1.0
    processor.has_quantization = True
    processor.model_parts = [model]
    processor.weight_updates = WeightUpdateMonitor(enabled=True)

    with processor.observe_weight_updates(True):
        optim.step()
    processor.log(1, 0.5, 0.5, 1.0, extra_metrics={"n_tokens_seen": 3})

    assert logged["weight_update/linear/Linear"] == pytest.approx(
        relative_l2_distance(torch.ones(2, 2), model.weight)
    )
    assert logged["n_tokens_seen"] == 3
    assert logged["grad_norm"] == 1.0
    assert logged["fp4_failure/A_sub_ulp_fraction"] == pytest.approx(0.0)
    assert logged["fp4_failure/B_sign_flip_fraction"] == pytest.approx(0.0)
    assert processor.weight_updates.consume() == {}


def test_storage_ulp_matches_bfloat16_power_of_two_gap():
    weight = torch.tensor([256.0], dtype=torch.bfloat16)
    # 256 = 2^8 and BF16 has 7 mantissa bits, so the gap above 256 is 2.
    assert float(_storage_ulp(weight)) == pytest.approx(2.0)


def test_sub_ulp_fraction_is_one_when_bf16_add_rounds_away():
    layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.fill_(256)
    layer.weight = nn.Parameter(layer.weight.bfloat16())
    # Half an ulp of 256 in BF16 is 1. A step of 0.5 cannot change the stored value.
    layer.weight.grad = torch.ones_like(layer.weight)
    optim = torch.optim.SGD(layer.parameters(), lr=0.5)
    monitor = WeightUpdateMonitor(enabled=True)
    with monitor.observe([layer], active=True):
        optim.step()
    extra = monitor.consume()
    assert extra["fp4_failure/A_sub_ulp_fraction"] == pytest.approx(1.0)
    assert extra["fp4_failure/A_max_sub_ulp_fraction"] == pytest.approx(1.0)
    assert "fp4_failure/A_median_update_over_ulp" not in extra
    assert torch.equal(layer.weight, torch.full_like(layer.weight, 256))


def test_sign_flip_exact_zero_and_outlier_fractions():
    layer = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.25)
        layer.weight.view(-1)[0] = 100.0
    layer.weight.grad = torch.ones_like(layer.weight)
    optim = torch.optim.SGD(layer.parameters(), lr=1.0)
    before = layer.weight.detach().clone()
    monitor = WeightUpdateMonitor(enabled=True)
    with monitor.observe([layer], active=True):
        optim.step()
    extra = monitor.consume()
    after = layer.weight.detach()
    sign_flip = ((before > 0) & (after < 0)) | ((before < 0) & (after > 0))
    assert extra["fp4_failure/B_sign_flip_fraction"] == pytest.approx(
        float(sign_flip.float().mean())
    )
    assert extra["fp4_failure/B_sign_flip_fraction"] > 0.9
    assert extra["fp4_failure/E_max_over_rms"] > 6.0
    assert extra["fp4_failure/E_outlier_fraction"] > 0.0
    assert "fp4_failure/E_update_sign_reversal_fraction" not in extra

    zeros = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        zeros.weight.zero_()
    zeros.weight.grad = torch.zeros_like(zeros.weight)
    optim = torch.optim.SGD(zeros.parameters(), lr=0.1)
    monitor = WeightUpdateMonitor(enabled=True)
    with monitor.observe([zeros], active=True):
        optim.step()
    extra = monitor.consume()
    assert extra["fp4_failure/B_exact_zero_fraction"] == pytest.approx(1.0)
    assert extra["fp4_failure/B_near_zero_fraction"] == pytest.approx(1.0)


def test_update_sign_reversal_uses_the_previous_logged_step():
    layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    optim = torch.optim.SGD(layer.parameters(), lr=0.1)
    monitor = WeightUpdateMonitor(enabled=True)

    layer.weight.grad = torch.ones_like(layer.weight)
    with monitor.observe([layer], active=True):
        optim.step()
    assert "fp4_failure/E_update_sign_reversal_fraction" not in monitor.consume()

    layer.weight.grad = -torch.ones_like(layer.weight)
    with monitor.observe([layer], active=True):
        optim.step()
    extra = monitor.consume()
    assert extra["fp4_failure/E_update_sign_reversal_fraction"] == pytest.approx(1.0)


def test_head_row_mean_drift_is_logged_for_the_output_projection():
    class _Output(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.final_layer = _Output()
            self.body = nn.Linear(4, 4, bias=False)

    model = _Model()
    with torch.no_grad():
        model.final_layer.linear.weight.fill_(2.0)
        model.body.weight.fill_(2.0)
    model.final_layer.linear.weight.grad = -torch.ones_like(model.final_layer.linear.weight)
    model.body.weight.grad = torch.zeros_like(model.body.weight)
    optim = torch.optim.SGD(
        [model.final_layer.linear.weight, model.body.weight], lr=1.0
    )
    monitor = WeightUpdateMonitor(enabled=True)
    with monitor.observe([model], active=True):
        optim.step()
    extra = monitor.consume()
    # Every output row moved from 2 to 3, so the row-mean vector grew by half.
    assert extra["fp4_failure/C_head_row_mean_relative_drift"] == pytest.approx(0.5)
    assert extra["fp4_failure/C_head_row_mean_rms"] == pytest.approx(3.0)
    assert extra["fp4_failure/C_head_update_mean_over_rms"] == pytest.approx(1.0)
    assert _is_head("weight_update/linear/final_layer.linear")
    assert _is_head("weight_update/linear/lm_head")
    assert not _is_head("weight_update/linear/body")


def test_failure_record_counts_a_sub_ulp_step_and_summarizes_heads_only():
    before = torch.tensor([[256.0]], dtype=torch.bfloat16)
    after = before.clone()
    record, sign = _failure_record(before, after, prev_sign=None)
    assert record["sub_ulp"] == pytest.approx(1.0)
    assert record["n_moved"] == pytest.approx(0.0)
    assert torch.equal(sign, torch.zeros(1, 1, dtype=torch.int8))

    summary = _summarize_failures(
        {
            "weight_update/linear/body": {
                **record,
                "row_mean_rel_drift": 9.0,
            },
            "weight_update/linear/lm_head": {
                **record,
                "numel": 4.0,
                "sub_ulp": 0.0,
                "row_mean_rms": 1.5,
                "row_mean_rel_drift": 0.25,
                "update_mean_over_rms": 0.4,
            },
        }
    )
    assert summary["fp4_failure/C_head_row_mean_relative_drift"] == pytest.approx(0.25)
    assert summary["fp4_failure/A_sub_ulp_fraction"] == pytest.approx(1.0 / 5.0)
    assert "fp4_failure/E_update_sign_reversal_fraction" not in summary
