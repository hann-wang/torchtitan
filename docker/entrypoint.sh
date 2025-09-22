#!/bin/bash

set -ex

cd /usr/local/src/torchtitan

CONFIG_FILE="./config.toml"
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
# Set cluster ENV
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export LOG_RANK=${LOG_RANK:-0}
export PYTORCH_ALLOC_CONF="expandable_segments:True"

cat > $CONFIG_FILE <<EOL
[job]
dump_folder = "./deepseek-v3-lite20b-mi300-bf16-outputs"
description = "DeepSeek-V3 16B model training"
print_args = false

[profiling]
enable_profiling = false
save_traces_folder = "profile_trace"
profile_freq = 10
enable_memory_snapshot = false
save_memory_snapshot_folder = "memory_snapshot"

[metrics]
log_freq = 1
disable_color_printing = false
enable_tensorboard = true
save_tb_folder = "tb"
enable_wandb = false

[model]
name = "deepseek_v3"
flavor = "20B"
hf_assets_path = "./assets/hf/DeepSeek-V3-Base"
# converters = ["float8"]
# converters = ["mx"]

[optimizer]
name = "AdamW"
lr = 2.2e-4
eps = 1e-8

[lr_scheduler]
warmup_steps = 600  # lr scheduler warm up, normally 20% of the train steps
decay_ratio = 0.8  # lr scheduler decay ratio, 80% of the train steps
decay_type = "cosine"
min_lr_factor = 0.1

[training]
local_batch_size = 1
seq_len = 4096
max_norm = 1.0  # grad norm clipping
steps = 3000
dataset = "c4_test"  # supported datasets: c4_test (2K), c4 (177M)
seed = 1234

[parallelism]
data_parallel_replicate_degree = $NNODES
data_parallel_shard_degree = -1
fsdp_reshard_after_forward = "default" # default / never / always
tensor_parallel_degree = 1
enable_async_tensor_parallel = false
pipeline_parallel_degree = 1
pipeline_parallel_schedule = "Interleaved1F1B"
expert_parallel_degree = 4
expert_tensor_parallel_degree = 1

[checkpoint]
enable = false
folder = "checkpoint"
interval = 10
last_save_model_only = true
export_dtype = "float32"
async_mode = "disabled"  # ["disabled", "async", "async_with_pinned_mem]"

[activation_checkpoint]
mode = "none"  # ["none", "selective", "full"]
selective_ac_option = "1"  # 'int' = ac every positive int layer or 'op', ac based on ops policy

[compile]
enable=true
components = ["loss"] # ["model", "loss"]

[float8]
enable_fsdp_float8_all_gather = false
precompute_float8_dynamic_scale_for_fsdp = false
filter_fqns = ["output", "router.gate", "attention.wq", "attention.wq_a", "attention.wq_b", "attention.wkv_a", "attention.wkv_b", "attention.wo"]
moe_fqns_prototype = ["experts"]
recipe_name = "blockwise"
enable_fp8_fa = true
enable_fp8_gmm = true
enable_fp8_linear = true

[mx]
filter_fqns = ["output", "router.gate", "attention.wq", "attention.wq_a", "attention.wq_b", "attention.wkv_a", "attention.wkv_b", "attention.wo"]
recipe_name = "mxfp4_1d1d"
enable_mxfp4_fa = false
enable_mxfp4_gmm = true
enable_mxfp4_linear = true
use_sr_grad = false
EOL

PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF \
torchrun \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  --local-ranks-filter ${LOG_RANK} \
  --role rank --tee 3 \
  -m $TRAIN_FILE --job.config_file ${CONFIG_FILE} "$@"
