#!/bin/bash

set -ex

cd /usr/local/src/torchtitan

CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_20b.toml"
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

# Set cluster ENV
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export LOG_RANK=${LOG_RANK:-0}
export PYTORCH_ALLOC_CONF="expandable_segments:True"

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
