from typing import Optional
import torch
from torch import nn

# torchao
from torchao.float8.config import Float8LinearConfig, ScalingType, e4m3_dtype, e5m2_dtype
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig, ScaledMMConfig
from torchao.float8.float8_utils import (
    amax_history_to_scale,
    tensor_to_amax,
    tensor_to_scale,
    to_fp8_saturated,
)
from torchao.float8.float8_scaling_utils import (
    _maybe_initialize_amaxes_scales_for_float8_cast,
    hp_tensor_to_float8_delayed, hp_tensor_to_float8_dynamic,
    hp_tensor_to_float8_static, hp_tensor_and_scale_to_float8
)


@torch._dynamo.allow_in_graph
class ToFloat8(torch.autograd.Function):
    """
    A differentiable conversion from fp8.
    * forward: convert from float8 to high precision
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(ctx, tensor, fp8_amax_input, fp8_amax_history_input,
                fp8_scale_input, scale_fn_name, is_amax_initialized,
                linear_mm_config, input_target_dtype):
        _maybe_initialize_amaxes_scales_for_float8_cast(
            tensor,
            fp8_amax_input,
            fp8_amax_history_input,
            fp8_scale_input,
            scale_fn_name,
            input_target_dtype,
            is_amax_initialized,
            reduce_amax=True,
        )
        input_fp8 = hp_tensor_to_float8_delayed(
            tensor,
            fp8_scale_input,
            input_target_dtype,
            fp8_amax_input,
            linear_mm_config=linear_mm_config,
            gemm_input_role=GemmInputRole.INPUT,
        )

        return input_fp8._data

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None, None, None, None


@torch._dynamo.allow_in_graph
class FromFloat8(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to float8_e5m2 with delayed scaling, initialize if needed
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        fp8_amax_grad_output,
        fp8_amax_history_grad_output,
        fp8_scale_grad_output,
        scale_fn_name,
        is_amax_initialized,
        linear_mm_config: LinearMMConfig,
        target_dtype: torch.dtype,
    ):
        ctx.save_for_backward(fp8_amax_grad_output,
                              fp8_amax_history_grad_output,
                              fp8_scale_grad_output)
        ctx.scale_fn_name = scale_fn_name
        ctx.is_amax_initialized = is_amax_initialized
        ctx.linear_mm_config = linear_mm_config
        ctx.target_dtype = target_dtype
        return tensor

    @staticmethod
    def backward(ctx, go):
        (
            fp8_amax_grad_output,
            fp8_amax_history_grad_output,
            fp8_scale_grad_output,
        ) = ctx.saved_tensors
        scale_fn_name = ctx.scale_fn_name
        is_amax_initialized = ctx.is_amax_initialized

        _maybe_initialize_amaxes_scales_for_float8_cast(
            go,
            fp8_amax_grad_output,
            fp8_amax_history_grad_output,
            fp8_scale_grad_output,
            scale_fn_name,
            ctx.target_dtype,
            is_amax_initialized,
            reduce_amax=True,
        )
        fp8_amax_grad_output.fill_(tensor_to_amax(go))

        res = hp_tensor_and_scale_to_float8(
            go,
            fp8_scale_grad_output,
            ctx.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
        )
        empty_grads = None, None, None, None, None, None, None
        return res, *empty_grads
        # return go.clone(), *empty_grads

class Spectator(nn.Module):

    def __init__(self, watch_fw=False, watch_bw=False):
        super(Spectator, self).__init__()
        self.always_float32_buffers = set()
        self.watch_fw = watch_fw
        self.watch_bw = watch_bw
        self.input_target_dtype = e4m3_dtype
        self.weight_target_dtype = e4m3_dtype
        self.grad_output_target_dtype = e5m2_dtype

    def forward(self, x):
        # this will record the input scale in forward
        if self.watch_fw:
            x = self.cast_input_to_float8(x)

        # this will record the grad scale in backward
        if self.watch_bw:
            fp8_output = self.cast_output_to_float8_in_bw(x)
            x = fp8_output

        return x

    def get_input_scale(self):
        return self.fp8_scale_input

    def get_grad_scale(self):
        return self.fp8_scale_grad_output

    def cast_input_to_float8(self, input: torch.Tensor) -> torch.Tensor:
        is_amax_initialized = self.is_amax_initialized
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)

        input_fp8 = ToFloat8.apply(
            input, self.fp8_amax_input, self.fp8_amax_history_input,
            self.fp8_scale_input,
            self.config.delayed_scaling_config.scale_fn_name,
            self.is_amax_initialized, self.linear_mm_config,
            self.input_target_dtype)

        return input_fp8

    def register_always_float32_buffer(self,
                                       name: str,
                                       tensor: Optional[torch.Tensor],
                                       persistent: bool = True) -> None:

        self.register_buffer(name=name, tensor=tensor, persistent=persistent)
        self.always_float32_buffers.add(name)

    def cast_output_to_float8_in_bw(self,
                                    output: torch.Tensor) -> torch.Tensor:
        scale_fn_name = self.config.delayed_scaling_config.scale_fn_name
        output = FromFloat8.apply(
            output,
            self.fp8_amax_grad_output,
            self.fp8_amax_history_grad_output,
            self.fp8_scale_grad_output,
            scale_fn_name,
            self.is_amax_initialized,
            self.linear_mm_config,
            self.grad_output_target_dtype,
        )
        return output

    def create_buffers(self, config, device='cpu'):
        """
        this function should be called after the layer was initilized
        """
        # Default values for history buffers, see above TODO
        self.config = config
        self.is_amax_initialized = not config.enable_amax_init

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

        history_len = self.config.delayed_scaling_config.history_len
        default_input = torch.finfo(
            self.input_target_dtype).max
        default_weight = torch.finfo(
            self.weight_target_dtype).max
        default_grad_output = torch.finfo(
            self.grad_output_target_dtype).max

        # Note: for now, create all the buffers if any are needed, to postpone
        # the work to make the scale and amax syncing and history calculation
        # handle a heterogeneous setup. We can do that work later if benchmarks
        # show it is worth doing.
        self.register_always_float32_buffer(
            "fp8_amax_input", torch.tensor([default_input], device=device))
        self.register_always_float32_buffer(
            "fp8_amax_history_input", torch.zeros(history_len, device=device))
        self.register_always_float32_buffer("fp8_scale_input",
                                            torch.tensor([1.0], device=device))
        self.register_always_float32_buffer(
            "fp8_amax_weight", torch.tensor([default_weight], device=device))
        self.register_always_float32_buffer(
            "fp8_amax_history_weight", torch.zeros(history_len, device=device))
        self.register_always_float32_buffer("fp8_scale_weight",
                                            torch.tensor([1.0], device=device))
        self.register_always_float32_buffer(
            "fp8_amax_grad_output",
            torch.tensor([default_grad_output], device=device),
        )
        self.register_always_float32_buffer(
            "fp8_amax_history_grad_output",
            torch.zeros(history_len, device=device))
        self.register_always_float32_buffer("fp8_scale_grad_output",
                                            torch.tensor([1.0], device=device))
