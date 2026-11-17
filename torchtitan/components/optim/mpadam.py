# mypy: allow-untyped-defs
"""MPAdam: an Adam variant tailored for low-precision (e.g. bf16/fp8) weights.

Reference algorithm: https://github.com/Anima-Lab/M-Adam-Low-precision-training
(``optimizer_torch/m_adam.py``).

This implementation intentionally departs from the reference in one
important way: **it does not manage its own learning-rate schedule**. The
reference optimizer bakes a warmup/cosine/logcosine schedule for the
exponent learning rate directly into ``step()`` (driven by an internal
step counter ``t``). Here, following the convention used by
:class:`torch.optim.Adam`/:class:`torch.optim.AdamW`, ``lr`` is simply read
from ``param_group["lr"]`` on every step and is expected to be driven by a
``torch.optim.lr_scheduler`` (or any external code that mutates the param
group) instead. The exponent step uses ``lr * lr_e_ratio`` on every
step, so the same scheduler scales both updates.

Structurally this module mirrors the reference repo's own simplified
``optimizer_torch/adam.py`` (a from-scratch, always-fp32-math, single
for-loop Adam/AdamW used there as a baseline) rather than the full
``torch.optim.Adam``: there is no ``foreach``/``fused``/``capturable``/
``differentiable``/``amsgrad`` support, just a straightforward per-parameter
loop. Argument validation, the ``defaults`` dict, lazy per-parameter state
initialization, and ``@torch.no_grad() def step(self, closure=None)`` all
follow the same conventions as PyTorch's built-in optimizers so ``MPAdam``
is a drop-in swap wherever ``Adam``/``AdamW`` is used.
"""

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch import Tensor

from .optimizer import BaseOptimizer

__all__ = ["MPAdam"]


def _rms(x: Tensor) -> Tensor:
    return x.pow(2).mean().sqrt()


class MPAdam(BaseOptimizer):
    """Adam on a mantissa/exponent parameterization for low-precision weights.

    MPAdam decomposes each fp32 view of a weight into a mantissa/exponent
    pair via :func:`torch.frexp`, ``w = m * 2**e`` with ``m`` in
    ``[0.5, 1)``, and performs two coupled Adam-style updates every step:

    * an **exponent update** that adapts ``e`` from an Adam-normalized
      gradient of ``e`` (``d(loss)/de = grad * w * ln(2)``), so the working
      scale of a parameter can track gradient signal even when ``w`` itself
      is stored in a narrow low-precision format (e.g. bf16/fp8);
    * a **mantissa/weight update** that is an ordinary decoupled
      weight-decay Adam (i.e. AdamW) step computed on the reconstructed
      fp32 weight, then translated back into a mantissa delta at the *new*
      exponent, so the low-precision rounding introduced by casting back to
      the parameter's dtype happens at the right scale.

    ``lr`` is read from ``param_group["lr"]`` on every :meth:`step`, so any
    ``torch.optim.lr_scheduler`` drives it from the outside. The exponent
    learning rate is always ``lr * lr_e_ratio``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizer.Config):
        lr: float = 1e-3
        """Mantissa/weight learning rate. Read from the param group every step."""

        betas: tuple[float, float] = (0.9, 0.999)
        """Coefficients for the running averages of the gradient and its square.

        Shared by the mantissa and exponent updates.
        """

        eps: float = 1e-8
        """Term added to the denominator to improve numerical stability."""

        weight_decay: float = 0.0
        """Decoupled (AdamW-style) weight decay on the reconstructed fp32 weight."""

        lr_e_ratio: float = 1.0
        """Exponent learning rate divided by ``lr``.

        Every step uses ``lr * lr_e_ratio`` as the exponent step size, so an
        external schedule on ``lr`` moves both updates together.
        """

        weight_decay_e: float = 0.0
        """Decoupled weight decay on the exponent, pulling ``|w|`` back towards 1."""

        p_scale: float = 3.0
        """Multiplier on a parameter's initial RMS for the absolute-value clamp.

        Used when ``abs_clamp`` is true. ``abs_max`` is fixed once at
        initialization from this scale and ``abs_clamp_floor``.
        """

        g_bound: float = 20.0
        """Clamp bound applied to the normalized exponent gradient."""

        abs_clamp: bool = False
        """Clamp updated weights to ``[-abs_max, abs_max]``."""

        abs_clamp_floor: float = 0.0
        """Minimum value for ``abs_max``."""

        clip_e_final: bool = False
        """Clamp the exponent to ``[e_final_min, e_final_max]`` every step."""

        e_final_min: float = -60.0
        """Lower exponent clamp bound."""

        e_final_max: float = 60.0
        """Upper exponent clamp bound."""

        use_de_step_cap: bool = True
        """Clamp the per-step exponent delta to ``[-de_step_cap, de_step_cap]``."""

        de_step_cap: float = 0.5
        """Exponent per-step delta clamp bound."""

        maximize: bool = False
        """Maximize the objective with respect to the params, instead of minimizing."""

        def __post_init__(self) -> None:
            if not 0.0 <= self.lr:
                raise ValueError(f"Invalid learning rate: {self.lr}")
            if not 0.0 <= self.lr_e_ratio:
                raise ValueError(
                    f"Invalid exponent learning-rate ratio: {self.lr_e_ratio}"
                )
            if not 0.0 <= self.eps:
                raise ValueError(f"Invalid epsilon value: {self.eps}")
            if not 0.0 <= self.betas[0] < 1.0:
                raise ValueError(
                    f"Invalid beta parameter at index 0: {self.betas[0]}"
                )
            if not 0.0 <= self.betas[1] < 1.0:
                raise ValueError(
                    f"Invalid beta parameter at index 1: {self.betas[1]}"
                )
            if not 0.0 <= self.weight_decay:
                raise ValueError(f"Invalid weight_decay value: {self.weight_decay}")
            if not 0.0 <= self.weight_decay_e:
                raise ValueError(
                    f"Invalid weight_decay_e value: {self.weight_decay_e}"
                )
            if not 0.0 < self.p_scale:
                raise ValueError(f"Invalid p_scale value: {self.p_scale}")
            if not 0.0 < self.g_bound:
                raise ValueError(f"Invalid g_bound value: {self.g_bound}")
            if not 0.0 <= self.abs_clamp_floor:
                raise ValueError(
                    f"Invalid abs_clamp_floor value: {self.abs_clamp_floor}"
                )
            if not self.e_final_min < self.e_final_max:
                raise ValueError(
                    f"e_final_min ({self.e_final_min}) must be < "
                    f"e_final_max ({self.e_final_max})"
                )
            if not 0.0 < self.de_step_cap:
                raise ValueError(f"Invalid de_step_cap value: {self.de_step_cap}")

    def __init__(
        self,
        config: Config,
        *,
        params: Iterable[dict[str, Any]] | Iterable[Tensor],
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=config.lr,
                lr_e_ratio=config.lr_e_ratio,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
                weight_decay_e=config.weight_decay_e,
                p_scale=config.p_scale,
                g_bound=config.g_bound,
                abs_clamp=config.abs_clamp,
                abs_clamp_floor=config.abs_clamp_floor,
                clip_e_final=config.clip_e_final,
                e_final_min=config.e_final_min,
                e_final_max=config.e_final_max,
                use_de_step_cap=config.use_de_step_cap,
                de_step_cap=config.de_step_cap,
                maximize=config.maximize,
            ),
        )

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("lr_e_ratio", 1.0)
            group.setdefault("weight_decay_e", 0.0)
            group.setdefault("p_scale", 3.0)
            group.setdefault("g_bound", 20.0)
            group.setdefault("abs_clamp", False)
            group.setdefault("abs_clamp_floor", 0.0)
            group.setdefault("clip_e_final", False)
            group.setdefault("e_final_min", -60.0)
            group.setdefault("e_final_max", 60.0)
            group.setdefault("use_de_step_cap", True)
            group.setdefault("de_step_cap", 0.5)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): a closure that reevaluates the
                model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd_m = group["weight_decay"]
            wd_e = group["weight_decay_e"]
            g_bound = group["g_bound"]
            abs_clamp = group["abs_clamp"]
            clip_e = group["clip_e_final"]
            e_min, e_max = group["e_final_min"], group["e_final_max"]
            cap_de = group["use_de_step_cap"]
            de_cap = group["de_step_cap"]
            maximize = group["maximize"]

            # `lr` is read fresh every step so an external LRScheduler
            # scales both the mantissa step and the exponent step
            # (`lr * lr_e_ratio`).
            lr_m = group["lr"]
            lr_e = lr_m * group["lr_e_ratio"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("MPAdam does not support sparse gradients")

                param_dtype = p.data.dtype
                grad = p.grad.data
                if maximize:
                    grad = -grad
                if grad.dtype != param_dtype:
                    grad = grad.to(param_dtype)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Momentum/variance buffers live in the parameter's own
                    # (possibly low-precision) dtype, matching the
                    # reference implementation: the low-precision rounding
                    # of these buffers is part of what MPAdam simulates.
                    state["mantissa_exp_avg"] = torch.zeros_like(
                        p.data, dtype=param_dtype
                    )
                    state["mantissa_exp_avg_sq"] = torch.zeros_like(
                        p.data, dtype=param_dtype
                    )
                    state["exponent_exp_avg_sq"] = torch.zeros_like(
                        p.data, dtype=param_dtype
                    )
                    init_rms = _rms(p.data.float()).item()
                    state["abs_max"] = max(
                        group["p_scale"] * (init_rms + 1e-12),
                        group["abs_clamp_floor"],
                    )

                mantissa_m = state["mantissa_exp_avg"]
                mantissa_v = state["mantissa_exp_avg_sq"]
                exponent_v = state["exponent_exp_avg_sq"]

                w_fp32 = p.data.float()
                grad_fp32 = grad.float()

                # ---- decompose weight: w = mantissa * 2**exponent -------
                mantissa, exponent = torch.frexp(w_fp32)
                exponent_used = (
                    torch.clamp(exponent, min=e_min, max=e_max)
                    if clip_e
                    else exponent
                )

                state["step"] += 1
                step = state["step"]

                # Defensive clamp so exp2() below can't overflow fp32
                # regardless of clip_e_final.
                exponent_safe = torch.clamp(exponent_used, min=-60.0, max=60.0).to(
                    w_fp32.dtype
                )
                w_reconstructed = mantissa * torch.exp2(exponent_safe)

                # ---- exponent update: Adam on d(loss)/de = g * w * ln2 --
                grad_exponent = grad_fp32 * w_reconstructed * math.log(2.0)
                grad_exponent = grad_exponent.clamp(-1e19, 1e19)

                exponent_v_fp32 = exponent_v.float()
                exponent_v_fp32.mul_(beta2).addcmul_(
                    grad_exponent, grad_exponent, value=1.0 - beta2
                )
                exponent_denom = (exponent_v_fp32 / (1.0 - beta2**step)).sqrt_().clamp_(
                    min=eps
                )
                grad_exponent_n = (grad_exponent / exponent_denom).clamp_(
                    -g_bound, g_bound
                )
                exponent_step = -lr_e * grad_exponent_n

                # Translate the exponent-space step into a bounded delta-e
                # via log1p, so one step can move |w| by at most ~1.75x.
                weight_floor = max(1e-8, float(_rms(w_fp32)) * 1e-6)
                abs_w = w_reconstructed.abs().clamp_min(weight_floor)
                ratio = (exponent_step / abs_w).clamp(min=-0.75 + 1e-6, max=0.75)
                delta_e = torch.log1p(ratio) / math.log(2.0)
                if cap_de:
                    delta_e = delta_e.clamp(min=-de_cap, max=de_cap)

                exponent_new = exponent_used.to(w_fp32.dtype) * (
                    1.0 - lr_e * wd_e
                ) + delta_e
                if clip_e:
                    exponent_new = torch.clamp(exponent_new, min=e_min, max=e_max)
                exponent_v.copy_(exponent_v_fp32.to(param_dtype))

                # ---- mantissa/weight update: decoupled AdamW step -------
                # Gradient w.r.t. the mantissa is grad * 2**e (chain rule
                # through w = m * 2**e). It is deliberately rounded to the
                # parameter's dtype (`.to(param_dtype)`) and rescaled back
                # by 2**-e so the momentum buffers see the same
                # quantization noise a genuinely low-precision optimizer
                # operating directly on the mantissa would.
                exp_scale = torch.exp2(-exponent_used.to(w_fp32.dtype))
                grad_mantissa = grad_fp32 * torch.exp2(exponent_used.to(w_fp32.dtype))
                grad_mantissa_scaled = grad_mantissa.to(param_dtype) * exp_scale

                mantissa_m.mul_(beta1).add_(grad_mantissa_scaled, alpha=1.0 - beta1)
                mantissa_v.mul_(beta2).addcmul_(
                    grad_mantissa_scaled, grad_mantissa_scaled, value=1.0 - beta2
                )

                mantissa_m_hat = mantissa_m.float() / (1.0 - beta1**step)
                mantissa_v_hat = mantissa_v.float() / (1.0 - beta2**step)
                weight_step = -lr_m * mantissa_m_hat / (mantissa_v_hat.sqrt() + eps)

                if wd_m != 0.0:
                    w_updated = w_fp32 * (1.0 - lr_m * wd_m) + weight_step
                else:
                    w_updated = w_fp32 + weight_step

                # Translate the fp32 weight delta back into a mantissa
                # delta at the *new* exponent, so the mantissa stays in
                # [0.5, 1) after re-normalization by exp2(exponent_new).
                delta_w = w_updated - w_fp32
                delta_mantissa = delta_w * torch.exp2(-exponent_new)
                mantissa_new = mantissa + delta_mantissa
                w_new = mantissa_new * torch.exp2(exponent_new)

                if abs_clamp:
                    w_new.clamp_(-state["abs_max"], state["abs_max"])

                p.data.copy_(w_new.to(param_dtype))

        return loss
