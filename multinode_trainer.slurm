#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --- This script is optimized for AWS with EFA
# --- adjust NCCL_BUFFSIZE if you encounter memory
# --- constraint issues or to tune for improved performance.
# ---

#SBATCH --job-name=dsv3_torchtitan_multi_node

#SBATCH --ntasks=2

#SBATCH --nodes=2

#SBATCH --gpus-per-task=8

#SBATCH --cpus-per-task=128

#SBATCH --partition=AIG_Models

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# # Enable for A100
# export FI_PROVIDER="efa"
# # Ensure that P2P is available
# # export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

# # on your cluster you might need these:
# # set the network interface
# export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
# export NCCL_BUFFSIZE=2097152
# #export TORCH_DIST_INIT_BARRIER=1
# export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

export LOG_RANK=${LOG_RANK:-0}

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/deepseek_v3/train_configs/deepseek_v3.toml"}

DOCKER_RUN_CMD=${DOCKER_RUN_CMD:-"docker run --rm -t -u 300177:110 \
    --ulimit core=-1  --privileged \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --workdir /workspace/fp8/torchtitan-fa-fp8 \
    --ipc=host --shm-size 8G \
    --network host \
    -e HOME="/home/hanwang2" \
    -e HF_HOME="/huggingface" \
    -e TORCH_EXTENSIONS_DIR="/home/hanwang2/.local/share/torch_extensions" \
    -e LOG_RANK="$LOG_RANK" \
    -e NCCL_DEBUG="$NCCL_DEBUG" \
    -e PYTHONFAULTHANDLER="$PYTHONFAULTHANDLER" \
    -e LOGLEVEL="$LOGLEVEL" \
    -v /home/Han.Wang@amd.com/workspace:/workspace \
    -v /home/Han.Wang@amd.com/docker-home:/home/hanwang2 \
    -v /home/Han.Wang@amd.com/huggingface:/huggingface \
    -v /home/amd-shared-home:/home/amd-shared-home \
    han-workspace:ubuntu24.04-pytorch2.8.0dev-rocm6.2.4"}

# check if docker image exists on each node
for node in "${nodes_array[@]}"; do
    if ! srun --nodes=1 --ntasks=1 -w "$node" docker image inspect han-workspace:ubuntu24.04-pytorch2.8.0dev-rocm6.2.4 > /dev/null 2>&1; then
        # load from local tar file
        echo "Docker image not found on $node, loading from tar file..."
        srun --nodes=1 --ntasks=1 -w "$node" docker load -i /home/Han.Wang@amd.com/workspace/docker/han-workspace.tar
    fi
done

# dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun $DOCKER_RUN_CMD torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ./torchtitan/train.py --job.config_file ${CONFIG_FILE}
# dcgmi profile --resume
