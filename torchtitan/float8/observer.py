from typing import Optional
import logging
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed._functional_collectives import AsyncCollectiveTensor, all_reduce

# torchao
from torchao.float8.config import Float8LinearConfig, ScalingType, e4m3_dtype, e5m2_dtype
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig, ScaledMMConfig
from torchao.float8.float8_utils import (
    amax_history_to_scale,
    tensor_to_amax,
    tensor_to_scale,
    to_fp8_saturated,
    amax_history_to_scale_stack,
)
from torchao.float8.float8_scaling_utils import (
    _maybe_initialize_amaxes_scales_for_float8_cast,
    hp_tensor_to_float8_delayed, hp_tensor_to_float8_dynamic,
    hp_tensor_to_float8_static, hp_tensor_and_scale_to_float8
)
from torchao.float8.float8_linear_utils import _update_history_stack

log = logging.getLogger(__name__)

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

class Observer(nn.Module):

    def __init__(self, watch_fw=False, watch_bw=False):
        super(Observer, self).__init__()
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
        default_weight = torch.finfo(self.weight_target_dtype).max
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


def get_float8_observers(model: torch.nn.Module):
    """Iterates through the model and returns all the float8 observers.
    Args:
        model (torch.nn.Module): The model to look for float8 observers in.
    """

    fp8_observers = [
        child for child in model.modules() if isinstance(child, Observer)
    ]
    if not torch.compiler.is_compiling():
        for layer in fp8_observers:
            for buf in layer.buffers():
                torch._dynamo.mark_static_address(buf, guard=True)
    return fp8_observers


@torch.no_grad()
def sync_observer_amax_and_scale_history(model: torch.nn.Module,
                                       fp8_observers=None) -> None:
    """
    adapted from torchao.float8.sync_float8_amax_and_scale_history
    """

    if fp8_observers is None:
        fp8_observers = get_float8_observers(model)

    if len(fp8_observers) == 0:
        log.warning(
            "Calling sync_float8_amax_and_scale_history on a module with no Float8Linear layers"
        )
        return

    def inner_func():

        # Loop over all fp8 observers and grab the needed tensors
        fp8_amax_input_tensor_list = [None] * len(fp8_observers)
        fp8_amax_grad_output_tensor_list = [None] * len(fp8_observers)

        fp8_input_amax_history_stack = [None] * len(fp8_observers)
        fp8_grad_output_amax_history_stack = [None] * len(fp8_observers)

        input_dtypes = set()
        grad_output_dtypes = set()
        scale_fn_recipes = set()

        for idx, child in enumerate(fp8_observers):
            fp8_amax_input_tensor_list[idx] = child.fp8_amax_input
            fp8_amax_grad_output_tensor_list[idx] = child.fp8_amax_grad_output

            fp8_input_amax_history_stack[idx] = child.fp8_amax_history_input
            fp8_grad_output_amax_history_stack[
                idx] = child.fp8_amax_history_grad_output

            input_dtypes.add(child.config.cast_config_input.target_dtype)
            grad_output_dtypes.add(
                child.config.cast_config_grad_output.target_dtype)
            scale_fn_recipes.add(
                child.config.delayed_scaling_config.scale_fn_name)

        (input_dtype, ) = input_dtypes
        (grad_output_dtype, ) = grad_output_dtypes

        if len(scale_fn_recipes) != 1:
            raise ValueError(
                f"All layers must have the same scale_fn recipe, got {scale_fn_recipes}"
            )
        scale_fn_recipe = next(iter(scale_fn_recipes))

        assert (len(fp8_amax_input_tensor_list) ==
                len(fp8_amax_grad_output_tensor_list)
                ), "Mismatched lengths of amax tensors."

        if dist.is_initialized():
            all_amax_tensors = torch.cat(fp8_amax_input_tensor_list +
                                         fp8_amax_grad_output_tensor_list)
            all_reduced_amax_tensor = all_reduce(
                all_amax_tensors, "MAX", list(range(dist.get_world_size())))
            if isinstance(all_reduced_amax_tensor, AsyncCollectiveTensor):
                all_reduced_amax_tensor = all_reduced_amax_tensor.wait()

            (
                reduced_fp8_amax_input_tensor,
                reduced_fp8_amax_grad_output_tensor,
            ) = torch.split(all_reduced_amax_tensor,
                            len(fp8_amax_input_tensor_list))

            for idx, child in enumerate(fp8_observers):
                child.fp8_amax_input.copy_(reduced_fp8_amax_input_tensor[idx])
                child.fp8_amax_grad_output.copy_(
                    reduced_fp8_amax_grad_output_tensor[idx])

        # We create two stacked tensor groups, one for the amax history and one for the current scales
        fp8_amax_input_tensors = torch.vstack(fp8_amax_input_tensor_list)
        fp8_amax_grad_output_tensors = torch.vstack(
            fp8_amax_grad_output_tensor_list)

        fp8_input_amax_history_stack = torch.vstack(
            fp8_input_amax_history_stack)
        fp8_grad_output_amax_history_stack = torch.vstack(
            fp8_grad_output_amax_history_stack)

        # Update the history stacks with the new amax values
        _update_history_stack(fp8_amax_input_tensors,
                              fp8_input_amax_history_stack)
        _update_history_stack(fp8_amax_grad_output_tensors,
                              fp8_grad_output_amax_history_stack)

        # Calculate the new scales from the updated history stacks
        new_input_scales = amax_history_to_scale_stack(
            fp8_input_amax_history_stack, input_dtype, scale_fn_recipe)
        new_grad_output_scales = amax_history_to_scale_stack(
            fp8_grad_output_amax_history_stack, grad_output_dtype,
            scale_fn_recipe)

        # Iterate through the layers and update the scales
        for idx, child in enumerate(fp8_observers):
            child.fp8_scale_input.copy_(new_input_scales[idx])
            child.fp8_scale_grad_output.copy_(new_grad_output_scales[idx])

            child.fp8_amax_history_input.copy_(
                fp8_input_amax_history_stack[idx])
            child.fp8_amax_history_grad_output.copy_(
                fp8_grad_output_amax_history_stack[idx])

    # This allows for the compile to succeed on the inner func and fail on the graph breaks
    # at the beginning and and of syncing
    inner_func()

    for child in fp8_observers:
        # Set a flag to signal that initialization is done
        child.is_amax_initialized = True
