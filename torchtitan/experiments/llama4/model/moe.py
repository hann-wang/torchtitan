# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Shard, Replicate
from torchtitan.experiments.kernels.triton_contiguous_group_gemm.cg_backward import (
    cg_grouped_gemm, )
from torchtitan.experiments.kernels.moe.token_dispatcher import (
    DefaultTokenDispatcher,
    TorchAllToAllTokenDispatcher,
)

from .args import TransformerModelArgs

USE_CG_GROUPED_GEMM = True
ALIGN_SIZE_M = 128

class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.use_grouped_mm = use_grouped_mm
        self.use_fp8 = False

    def forward(
        self,
        x: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor | list[int] | None = None,
    ) -> torch.Tensor:
        # TODO: keeping this for loop implementation for comparison
        #       and readability, will remove later
        if isinstance(self.w1, DTensor) and self.w1.placements == (Shard(0),) and self.w1.device_mesh.size() > 1:
            # expert parallel enabled
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
            experts_per_rank = self.num_experts // self.w1.device_mesh.size()
        else:
            # expert parallel disabled
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3
            experts_per_rank = self.num_experts

        if not self.use_grouped_mm:
            assert not self.use_fp8
            if num_local_tokens_per_expert is not None:
                # a tuple of tensors indexed by experts
                # each with shape (tokens_per_expert(varying), dim)
                x = torch.split(
                    x,
                    split_size_or_sections=num_local_tokens_per_expert,
                    dim=0,
                )
                out_experts_splits = []
                for expert_idx, x_expert in enumerate(x):
                    expert_idx = expert_idx % experts_per_rank
                    current_w1, current_w2, current_w3 = (
                        w1[expert_idx],
                        w2[expert_idx],
                        w3[expert_idx],
                    )
                    h = F.silu(torch.matmul(x_expert, current_w1))
                    h = h * torch.matmul(x_expert, current_w3)
                    h = torch.matmul(h, current_w2)
                    # h shape (tokens_per_expert(varying), dim)
                    out_experts_splits.append(h)
                out = torch.cat(out_experts_splits, dim=0)
            else:
                bs, slen, dim = x.shape
                x = x.reshape(1, bs * slen, dim)
                # x shape (num_experts, tokens_per_expert, dim)
                h = F.silu(torch.bmm(x, w1))
                h = h * torch.bmm(x, w3)
                # out shape (num_experts, tokens_per_expert, dim)
                out = torch.bmm(h, w2)
                out = out.reshape(bs, slen, dim)
            return out

        # grouped mm implementation
        if num_local_tokens_per_expert is not None:
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2

            assert (
                x.dtype == self.w1.dtype == self.w2.dtype == self.w3.dtype == torch.bfloat16
            ), "torch._grouped_mm only supports bf16 dtypes"

            # https://github.com/pytorch/pytorch/pull/150374
            # NOTE: torch._gouped_mm requires bf16 dtypes
            #       and shapes to be multiple of 8
            offsets = torch.cumsum(
                num_local_tokens_per_expert, dim=0, dtype=torch.int32
            )

            if USE_CG_GROUPED_GEMM:
                from torchtitan.experiments.deepseek_v3 import dsgemm_utils

                # Create indices from offsets without CPU-GPU sync
                m_indices = dsgemm_utils.create_indices_from_offsets_nosync(
                    offsets)
                gate_proj = cg_grouped_gemm(
                    x,
                    w1,
                    m_indices,
                    use_fp8=self.use_fp8,
                )
                up_proj = cg_grouped_gemm(
                    x,
                    w3,
                    m_indices,
                    use_fp8=self.use_fp8,
                )
                # Apply activation
                hidden_outputs = F.silu(gate_proj) * up_proj
                # Run the third GEMM (down projection)
                out = cg_grouped_gemm(
                    hidden_outputs,
                    w2,
                    m_indices,
                    use_fp8=self.use_fp8,
                )
            else:
                assert not self.use_fp8
                # FIXME: grouped_mm does not require padded m_sizes
                from .grouped_mm_utils import gmm
                num_local_tokens_per_expert_cpu = num_local_tokens_per_expert.to(
                    dtype=torch.int64, device="cpu")
                g = gmm(x.contiguous(), w1.contiguous(),
                        num_local_tokens_per_expert_cpu.contiguous())
                assert not torch.isnan(g).any(), "NaN detected in grouped mm gate projection"
                h = F.silu(g)
                h = h * gmm(x, w3, num_local_tokens_per_expert_cpu)
                out = gmm(h, w2, num_local_tokens_per_expert_cpu)
        else:
            bs, slen, dim = x.shape
            x = x.reshape(1, bs * slen, dim)
            assert not self.use_fp8
            # fall back to regular bmm between 3D tensors
            # x shape (num_experts, tokens_per_expert, dim)
            h = F.silu(torch.bmm(x, w1))
            h = h * torch.bmm(x, w3)
            # out shape (num_experts, tokens_per_expert, dim)
            out = torch.bmm(h, w2)
            out = out.reshape(bs, slen, dim)
        return out

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        use_sigmoid (bool): Whether to use sigmoid or softmax for router scores. Default is False.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        use_sigmoid: bool = False,
        norm_topk_prob: bool = False,
        routed_scaling_factor: float | None = None,
        topk_method: str = "noaux",
        n_group: int = 1,
        topk_group: int = 1,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_sigmoid = use_sigmoid
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_local_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        num_tokens = x.shape[0]
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.use_sigmoid:
            scores = torch.sigmoid(scores.to(torch.float32))
        else:
            scores = F.softmax(scores.to(torch.float32), dim=1)

        match self.topk_method:
            case "greedy":
                top_scores, selected_experts_indices = torch.topk(
                    scores,
                    k=self.top_k,
                    dim=1,
                    sorted=False,
                )
            case "noaux":
                _, selected_experts_indices = torch.topk(
                    scores + expert_bias,
                    k=self.top_k,
                    dim=1,
                    sorted=False,
                )
                # top scores shape (bs*slen, top_k)
                # NOTE: The expert_bias is only used for routing. The gating value
                #       top_scores is still derived from the original scores.
                top_scores = scores.gather(dim=1, index=selected_experts_indices)
            case "noaux_tc":
                assert self.num_experts % self.n_group == 0, (
                    "num_experts must be divisible by n_group for noaux_tc routing"
                )
                scores_for_choice = scores + expert_bias.unsqueeze(0)
                group_scores = (scores_for_choice.view(
                    num_tokens, self.n_group,
                    -1).topk(2, dim=-1)[0].sum(dim=-1))  # [n, n_group]
                group_idx = torch.topk(
                    group_scores, k=self.topk_group, dim=-1, sorted=False
                )[
                    1
                ]  # [n, top_k_group]
                group_mask = torch.zeros_like(group_scores)  # [n, n_group]
                group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
                score_mask = (group_mask.unsqueeze(-1).expand(
                    num_tokens, self.n_group,
                    self.num_experts // self.n_group).reshape(num_tokens,
                                                              -1))  # [n, e]
                tmp_scores = scores_for_choice.masked_fill(
                    ~score_mask.bool(), 0.0
                )  # [n, e]
                _, selected_experts_indices = torch.topk(
                    tmp_scores,
                    k=self.top_k,
                    dim=-1,
                    sorted=False,
                )
                top_scores = scores.gather(1, selected_experts_indices)
            case _:
                raise ValueError(f"Unknown topk method: {self.topk_method}")


        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        if self.routed_scaling_factor is not None:
            top_scores = top_scores * self.routed_scaling_factor

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_local_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )
        top_scores = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return top_scores, token_indices_experts_sorted, num_local_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):

    def __init__(
        self,
        model_args: TransformerModelArgs,
        scoring_before_experts: bool = True,
    ):
        super().__init__()
        self.scoring_before_experts = scoring_before_experts
        dim = model_args.dim
        hidden_dim = 4 * model_args.dim
        ffn_dim_multiplier = model_args.ffn_dim_multiplier
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        self.num_experts = model_args.num_experts
        self.topk_method = model_args.topk_method

        hidden_dim_denom = 1
        if model_args.auto_scale_hidden_dim:
            hidden_dim_denom = model_args.top_k + int(model_args.use_shared_expert)

        if model_args.auto_scale_hidden_dim:
            hidden_dim = int(hidden_dim / hidden_dim_denom)
        hidden_dim += -hidden_dim % model_args.multiple_of

        self.use_grouped_mm = model_args.use_grouped_mm
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=self.num_experts,
            use_grouped_mm=self.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=self.num_experts,
            top_k=model_args.top_k,
            topk_method=self.topk_method,
        )
        self.shared_expert = (
            GroupedExperts(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=1,
                use_grouped_mm=self.use_grouped_mm,
            )
            if model_args.use_shared_expert
            else None
        )

        self.token_dispatcher = DefaultTokenDispatcher(self.num_experts)

        # auxiliary-loss-free load balancing
        self.load_balance_coeff = model_args.load_balance_coeff
        # the fields below are defined even when load_balance_coeff is None
        # to make initialization and checkpointing code simpler
        if self.topk_method == "noaux_tc":
            # Changed from torch.empty to torch.rand to avoid non-even
            # distribution for runs without actual weigths
            self.register_parameter(
                "expert_bias",
                torch.nn.Parameter(
                    torch.randn(self.num_experts, dtype=torch.get_default_dtype())),
            )
        else:
            self.register_buffer(
                "expert_bias",
                torch.randn(self.num_experts, dtype=torch.get_default_dtype()),
                persistent=True,
            )

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(self.num_experts, dtype=torch.get_default_dtype()),
            persistent=True,
        )

        # NOTE: forward hook, forward pre hook, or backward pre hook
        #       would conflict with activation checkpointing
        if self.load_balance_coeff is not None and self.load_balance_coeff > 0:
            self.register_full_backward_hook(self._update_expert_bias)

    def _update_expert_bias(self, *_):
        expert_bias_delta = self.load_balance_coeff * torch.sign(
            self.tokens_per_expert.mean() - self.tokens_per_expert
        )
        expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
        self.expert_bias.add_(expert_bias_delta)

        self.tokens_per_expert.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape

        if isinstance(self.expert_bias, DTensor):
            assert self.expert_bias.placements == (Replicate(),)
            expert_bias = self.expert_bias.to_local()
        else:
            expert_bias = self.expert_bias

        # top_scores and selected_indices shape (bs*slen*top_k,)
        # num_local_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            num_local_tokens_per_expert,
        ) = self.router(x.reshape(bs * slen, dim), expert_bias)

        if self.topk_method == "noaux":
            # will be used to update the expert bias for load balancing
            self.tokens_per_expert += num_local_tokens_per_expert

        # shape (bs*slen*top_k, dim)
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices.clone(
            ),  # FIXME: avoid NaN in the backward pass, maybe changed by permute_indices, related to ROCm?
        )

        # TODO: Find a better place to initialize the token dispatcher.
        #       I tried putting it in PrepareModuleInputOutputWithParams._apply,
        #       but caused torch compiling isses
        if (isinstance(self.experts.w1, DTensor) and self.experts.w1.placements == (Shard(0),)):
            self.token_dispatcher = TorchAllToAllTokenDispatcher(
                num_experts=self.num_experts,
                ep_size=self.experts.w1.device_mesh.size(),
                ep_group=self.experts.w1.device_mesh.get_group(),
            )

        (
            gathered_tokens,
            gathered_top_scores,
            tokens_per_expert_group,
            input_splits,
            output_splits,
        ) = self.token_dispatcher.token_permutation(
            routed_input,
            top_scores,
            num_local_tokens_per_expert,
            self.training,
        )

        if self.scoring_before_experts:
            gathered_tokens = (gathered_tokens.to(torch.float32) *
                               gathered_top_scores.reshape(-1, 1)).to(x.dtype)

        if self.use_grouped_mm:
            # NOTE: In order to use torch._grouped_mm, we need to make sure
            # the number of tokens each expert gets is a multiple of 16.
            # The following kernel helps achieve this via padding, without
            # incurring synchronization between device and host.
            from torchtitan.experiments.kernels.moe.indices import (
                generate_permute_indices,
            )

            with torch.no_grad():
                (
                    permuted_indices,
                    tokens_per_expert_group,
                    _,
                ) = generate_permute_indices(
                    tokens_per_expert_group,
                    self.token_dispatcher.experts_per_rank,
                    self.token_dispatcher.ep_size,
                    gathered_tokens.shape[0] +
                    self.token_dispatcher.experts_per_rank * ALIGN_SIZE_M,
                    ALIGN_SIZE_M,
                )
            gathered_tokens_appended = torch.vstack(
                (gathered_tokens, gathered_tokens.new_zeros((dim))))
            buffer_shape = gathered_tokens_appended.shape
            gathered_tokens = gathered_tokens_appended[permuted_indices, :]

            gathered_top_scores = torch.cat(
                (gathered_top_scores, gathered_top_scores.new_zeros(1)))
            gathered_top_scores = gathered_top_scores[permuted_indices]
        else:
            # NOTE: this would incur a synchronization between device and host
            if tokens_per_expert_group is not None:
                tokens_per_expert_group = tokens_per_expert_group.tolist()

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(gathered_tokens, tokens_per_expert_group)
        if not self.scoring_before_experts:
            routed_output = (routed_output * gathered_top_scores.reshape(-1, 1)).to(x.dtype)

        if self.use_grouped_mm:
            gathered_tokens_buffer = routed_output.new_empty(buffer_shape)
            gathered_tokens_buffer[permuted_indices, :] = routed_output
            routed_output = gathered_tokens_buffer[:(buffer_shape[0] - 1), :]

        returned_tokens = self.token_dispatcher.token_unpermutation(
            routed_output, input_splits, output_splits, self.training)

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x).reshape(bs * slen, dim)
        else:
            out = x.new_zeros((bs * slen, dim))

        out = out.scatter_add(dim=0, index=token_indices, src=returned_tokens)
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)

        nn.init.normal_(self.expert_bias)
        nn.init.zeros_(self.tokens_per_expert)
