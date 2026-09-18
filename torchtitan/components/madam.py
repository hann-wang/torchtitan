# mypy: allow-untyped-defs
"""M-Adam: an Adam variant tailored for low-precision (e.g. bf16/fp8) weights.

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
group) instead. See :class:`MAdam` for how the second, exponent-only
learning rate (``lr_e``) is handled.

Structurally this module mirrors the reference repo's own simplified
``optimizer_torch/adam.py`` (a from-scratch, always-fp32-math, single
for-loop Adam/AdamW used there as a baseline) rather than the full
``torch.optim.Adam``: there is no ``foreach``/``fused``/``capturable``/
``differentiable``/``amsgrad`` support, just a straightforward per-parameter
loop. Argument validation, the ``defaults`` dict, lazy per-parameter state
initialization, and ``@torch.no_grad() def step(self, closure=None)`` all
follow the same conventions as PyTorch's built-in optimizers so ``MAdam``
is a drop-in swap wherever ``Adam``/``AdamW`` is used.
"""

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["MAdam"]


def _rms(x: Tensor) -> Tensor:
    return x.pow(2).mean().sqrt()


class MAdam(Optimizer):
    r"""Implements the M-Adam algorithm for low-precision weight training.

    M-Adam decomposes each fp32 view of a weight into a mantissa/exponent
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

    Unlike the reference implementation, ``MAdam`` does not schedule its
    own learning rate. As with :class:`torch.optim.Adam`, ``lr`` is read
    from ``param_group["lr"]`` on every :meth:`step`, so any
    ``torch.optim.lr_scheduler`` (or manual updates to the param group)
    drives it from the outside. The exponent-only learning rate ``lr_e``
    can either:

    * track ``lr`` at a fixed ratio (``tie_e_to_m=True``): ``lr_e`` is
      recomputed every step as ``lr * (lr_e / lr)`` using the ratio fixed
      at construction time, so a single external scheduler acting on
      ``lr`` scales both updates together; or
    * be read directly, and independently, from ``param_group["lr_e"]``
      (``tie_e_to_m=False``, the default) -- useful if you want to keep the
      exponent step size constant, or drive it with your own schedule.

    Args:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups.
        lr (float, optional): mantissa/weight learning rate (default: 1e-3).
            Read from ``param_group["lr"]`` every step.
        betas (Tuple[float, float], optional): coefficients used for
            computing running averages of gradient and its square, shared
            by the mantissa and exponent updates (default: ``(0.9, 0.999)``).
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8).
        weight_decay (float, optional): decoupled (AdamW-style) weight
            decay applied to the reconstructed fp32 weight in the mantissa
            update (default: 0).
        lr_e (float, optional): exponent learning rate (default: 1e-2).
            Read from ``param_group["lr_e"]`` every step when
            ``tie_e_to_m=False``; when ``tie_e_to_m=True`` this value only
            seeds the initial ``lr_e / lr`` ratio.
        tie_e_to_m (bool, optional): if ``True``, derive ``lr_e`` from
            ``lr`` every step via the ratio fixed at construction
            (default: ``False``).
        weight_decay_e (float, optional): decoupled weight decay applied to
            the exponent itself, pulling ``|w|`` back towards 1
            (default: 0).
        p_scale (float, optional): multiplier on a parameter's initial RMS
            used to set its absolute-value clamp bound when
            ``abs_clamp=True`` (default: 3.0).
        g_bound (float, optional): clamp bound applied to the normalized
            exponent gradient (default: 20.0).
        abs_clamp (bool, optional): if ``True``, clamp updated weights to
            ``[-abs_max, abs_max]``, where ``abs_max`` is fixed once at
            initialization from ``p_scale`` and ``abs_clamp_floor``
            (default: ``False``).
        abs_clamp_floor (float, optional): minimum value for ``abs_max``
            (default: 0).
        clip_e_final (bool, optional): if ``True``, clamp the exponent to
            ``[e_final_min, e_final_max]`` every step (default: ``False``).
        e_final_min (float, optional): lower exponent clamp bound
            (default: -60.0).
        e_final_max (float, optional): upper exponent clamp bound
            (default: 60.0).
        use_de_step_cap (bool, optional): if ``True``, clamp the per-step
            exponent delta to ``[-de_step_cap, de_step_cap]``
            (default: ``True``).
        de_step_cap (float, optional): exponent per-step delta clamp bound
            (default: 0.5).
        maximize (bool, optional): maximize the objective with respect to
            the params, instead of minimizing (default: ``False``).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        lr_e: float = 1e-2,
        tie_e_to_m: bool = False,
        weight_decay_e: float = 0.0,
        p_scale: float = 3.0,
        g_bound: float = 20.0,
        abs_clamp: bool = False,
        abs_clamp_floor: float = 0.0,
        clip_e_final: bool = False,
        e_final_min: float = -60.0,
        e_final_max: float = 60.0,
        use_de_step_cap: bool = True,
        de_step_cap: float = 0.5,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_e:
            raise ValueError(f"Invalid exponent learning rate: {lr_e}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= weight_decay_e:
            raise ValueError(f"Invalid weight_decay_e value: {weight_decay_e}")
        if not 0.0 < p_scale:
            raise ValueError(f"Invalid p_scale value: {p_scale}")
        if not 0.0 < g_bound:
            raise ValueError(f"Invalid g_bound value: {g_bound}")
        if not 0.0 <= abs_clamp_floor:
            raise ValueError(f"Invalid abs_clamp_floor value: {abs_clamp_floor}")
        if not e_final_min < e_final_max:
            raise ValueError(
                f"e_final_min ({e_final_min}) must be < e_final_max ({e_final_max})"
            )
        if not 0.0 < de_step_cap:
            raise ValueError(f"Invalid de_step_cap value: {de_step_cap}")

        # Fixed at construction time so `tie_e_to_m=True` can recompute
        # `lr_e = lr * lr_e_ratio` every step, automatically tracking
        # whatever schedule an external LRScheduler applies to `lr`.
        lr_e_ratio = lr_e / max(1e-20, lr)

        defaults = dict(
            lr=lr,
            lr_e=lr_e,
            lr_e_ratio=lr_e_ratio,
            tie_e_to_m=tie_e_to_m,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_e=weight_decay_e,
            p_scale=p_scale,
            g_bound=g_bound,
            abs_clamp=abs_clamp,
            abs_clamp_floor=abs_clamp_floor,
            clip_e_final=clip_e_final,
            e_final_min=e_final_min,
            e_final_max=e_final_max,
            use_de_step_cap=use_de_step_cap,
            de_step_cap=de_step_cap,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("lr_e", 1e-2)
            group.setdefault("lr_e_ratio", group["lr_e"] / max(1e-20, group["lr"]))
            group.setdefault("tie_e_to_m", False)
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

            # `lr`/`lr_e` are read fresh every step (rather than tracked
            # internally) so an external LRScheduler mutating
            # `param_group["lr"]` (and, optionally, `param_group["lr_e"]`)
            # is all that's needed to schedule this optimizer.
            lr_m = group["lr"]
            lr_e = lr_m * group["lr_e_ratio"] if group["tie_e_to_m"] else group["lr_e"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "MAdam does not support sparse gradients"
                    )

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
                    # of these buffers is part of what M-Adam simulates.
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
                exponent_safe = torch.clamp(
                    exponent_used, min=-60.0, max=60.0
                ).to(w_fp32.dtype)
                w_reconstructed = mantissa * torch.exp2(exponent_safe)

                # ---- exponent update: Adam on d(loss)/de = g * w * ln2 --
                grad_exponent = grad_fp32 * w_reconstructed * math.log(2.0)
                grad_exponent = grad_exponent.clamp(-1e19, 1e19)

                exponent_v_fp32 = exponent_v.float()
                exponent_v_fp32.mul_(beta2).addcmul_(
                    grad_exponent, grad_exponent, value=1.0 - beta2
                )
                exponent_denom = (
                    (exponent_v_fp32 / (1.0 - beta2**step)).sqrt_().clamp_(min=eps)
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

                exponent_new = (
                    exponent_used.to(w_fp32.dtype) * (1.0 - lr_e * wd_e) + delta_e
                )
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
                grad_mantissa = grad_fp32 * torch.exp2(
                    exponent_used.to(w_fp32.dtype)
                )
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
