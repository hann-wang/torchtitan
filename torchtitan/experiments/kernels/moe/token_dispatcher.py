from typing import Tuple, List, Optional
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed._functional_collectives import _resolve_group_name, RANK_TYPES, _maybe_wrap_tensor


@torch.compiler.allow_in_graph
class _FromTorchTensor(torch.autograd.Function):
    """
    _FromTorchTensor allows autograd to propagate from a normal Tensor to an
    AsyncCollectiveTensor.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,  # pyre-ignore[2]: Parameter must be annotated.
        input: torch.Tensor,
    ) -> torch.Tensor:
        return _maybe_wrap_tensor(input)

    @staticmethod
    def backward(
            ctx, grad_output: torch.Tensor
    ) -> torch.Tensor:  # type: ignore[override]
        return grad_output

def all_to_all_single_autograd(
    self: torch.Tensor,
    output_split_sizes: Optional[list[int]],
    input_split_sizes: Optional[list[int]],
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    Same as all_to_all_single but supports autograd.
    """
    if output_split_sizes is not None:
        assert all(
            isinstance(size, (int, torch.SymInt))
            for size in output_split_sizes), output_split_sizes
    if input_split_sizes is not None:
        assert all(
            isinstance(size, (int, torch.SymInt))
            for size in input_split_sizes), input_split_sizes

    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)
    if output_split_sizes is None or input_split_sizes is None:
        assert output_split_sizes is None and input_split_sizes is None, (
            "output_split_sizes and input_split_sizes must either be "
            "specified together or both set to None")
        output_split_sizes = [self.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes
    tensor = torch.ops._c10d_functional_autograd.all_to_all_single(  # type: ignore[attr-defined]
        self,
        output_split_sizes,
        input_split_sizes,
        group_name,
    )
    return _FromTorchTensor.apply(tensor)

class DefaultTokenDispatcher:

    def __init__(self, num_experts: int, ep_size: int = 1):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.experts_per_rank = num_experts // ep_size

    def token_permutation(
        self,
        routed_input: torch.Tensor,
        top_scores: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None,
               torch.Tensor | None]:
        return routed_input, top_scores, num_local_tokens_per_expert, None, None

    def token_unpermutation(
        self,
        routed_output: torch.Tensor,
        input_splits: torch.Tensor | None = None,
        output_splits: torch.Tensor | None = None,
        training: bool = True,
    ) -> torch.Tensor:
        return routed_output


class TorchAllToAllTokenDispatcher(DefaultTokenDispatcher):

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        ep_mesh,
    ):
        super().__init__(num_experts, ep_size)
        self.ep_mesh = ep_mesh

    def ep_group(self):
        return self.ep_mesh.get_group()

    def token_permutation(
        self,
        routed_input: torch.Tensor,
        top_scores: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int] | None,
               List[int] | None]:
        dim = routed_input.shape[-1]
        with torch.no_grad():
            tokens_per_expert_group = num_local_tokens_per_expert.new_empty(
                num_local_tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group,
                                   num_local_tokens_per_expert,
                                   group=self.ep_group())
            input_splits = num_local_tokens_per_expert.view(
                self.ep_size, -1).sum(dim=1).tolist()
            output_splits = tokens_per_expert_group.view(
                self.ep_size, -1).sum(dim=1).tolist()
        torch._check(sum(input_splits) == routed_input.shape[0])
        if training:
            gathered_tokens = all_to_all_single_autograd(
                routed_input,
                output_splits,
                input_splits,
                self.ep_group(),
            )
            gathered_top_scores = all_to_all_single_autograd(
                top_scores,
                output_splits,
                input_splits,
                self.ep_group(),
            )
        else:
            # TODO: unify with all_to_all_single_autograd after
            # https://github.com/pytorch/pytorch/issues/154370 is resolved
            gathered_num_tokens = sum(output_splits)
            gathered_tokens = routed_input.new_empty(
                (gathered_num_tokens, dim))
            dist.all_to_all_single(
                gathered_tokens,
                routed_input,
                output_splits,
                input_splits,
                group=self.ep_group(),
            )
            gathered_top_scores = top_scores.new_empty(gathered_num_tokens, )
            dist.all_to_all_single(
                gathered_top_scores,
                top_scores,
                output_splits,
                input_splits,
                group=self.ep_group(),
            )
        return gathered_tokens, gathered_top_scores, tokens_per_expert_group, input_splits, output_splits

    def token_unpermutation(
        self,
        routed_output: torch.Tensor,
        input_splits: List[int] | None = None,
        output_splits: List[int] | None = None,
        training: bool = True,
    ) -> torch.Tensor:
        dim = routed_output.shape[-1]
        if training:
            returned_tokens = all_to_all_single_autograd(
                routed_output,
                input_splits,
                output_splits,
                self.ep_group(),
            )
        else:
            # TODO: unify with all_to_all_single_autograd after
            # https://github.com/pytorch/pytorch/issues/154370 is resolved
            returned_tokens = routed_output.new_empty(
                (sum(input_splits), dim))
            dist.all_to_all_single(
                returned_tokens,
                routed_output,
                input_splits,
                output_splits,
                group=self.ep_group(),
            )
        return returned_tokens
