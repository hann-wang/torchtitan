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

#SBATCH --ntasks=4

#SBATCH --nodes=4

#SBATCH --gpus-per-task=8

#SBATCH --cpus-per-task=128

#SBATCH --partition=amd-tw

nodes_array=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $NF}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export LOG_RANK=${LOG_RANK:-0}

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

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eno0,eth0,eth1,eth2,eth3,eth4,eth5,eth6,eth7"
export NCCL_IB_HCA="rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1"
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_BUFFSIZE=2097152
# #export TORCH_DIST_INIT_BARRIER=1
# export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

# ----------------- AMD-specific GPU optimizations -----------------
# Enable system DMA engine (SDMA) on AMD GPUs for better IO throughput
export HSA_ENABLE_SDMA=1
# Prevent scratch memory from being reclaimed to stabilize large memory usage patterns (e.g., KV cache, MoE experts)
# NOTE: Must disable scratch reclaim to avoid MoE training crash on AMD GPUs
# Setting this to 0 prevents core dumps when using Mixture-of-Experts (MoE) models
export HSA_NO_SCRATCH_RECLAIM=0
# Disable MSCCL (RCCL multi-connection feature) for better stability
export RCCL_MSCCL_ENABLE=0

# ----------------- Performance tuning -----------------

# Disable NCCL internal checks to reduce overhead
export NCCL_CHECKS_DISABLE=1

# Limit GPU hardware queues to 2 for performance stability
export GPU_MAX_HW_QUEUES=2

# Limit max CUDA device connections to reduce PCIe traffic
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# Prioritize NCCL communication for PyTorch for higher throughput
export TORCH_NCCL_HIGH_PRIORITY=1

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/deepseek_v3/train_configs/deepseek_200b.toml"}
IMAGE_NAME="han-workspace:ubuntu24.04-pytorch2.8.0dev-rocm6.4"
IMAGE_FILE="/home/amd/cache/hanwang2/docker/han-workspace.tar"

DOCKER_RUN_CMD=${DOCKER_RUN_CMD:-"docker run --rm -t -u 1003:110 \
    --ulimit core=-1  --privileged \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --group-add video \
    --group-add render \
    --workdir /workspace/fp8/torchtitan-fa-fp8 \
    --ipc=host --shm-size 8G \
    --network host \
    -e HOME=/home/hanwang2 \
    -e HF_HOME=/huggingface \
    -e HF_HUB_OFFLINE=1 \
    -e TORCH_EXTENSIONS_DIR=/home/hanwang2/.local/share/torch_extensions \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e LOG_RANK=${LOG_RANK} \
    -e NCCL_DEBUG=${NCCL_DEBUG} \
    -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
    -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
    -e NCCL_IB_HCA=${NCCL_IB_HCA} \
    -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
    -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC} \
    -e HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA} \
    -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM} \
    -e RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE} \
    -e NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE} \
    -e GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES} \
    -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \
    -e TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY} \
    -e NCCL_BUFFSIZE=${NCCL_BUFFSIZE} \
    -e PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER} \
    -e LOGLEVEL=${LOGLEVEL} \
    -v /home/amd/cache/hanwang2/workspace:/workspace \
    -v /home/amd/cache/hanwang2/docker-home-fp8:/home/hanwang2 \
    -v /home/amd/cache/huggingface:/huggingface \
    -v /home/amd/cache:/home/amd/cache \
    -v /home/amd/cache/hanwang2/tmp:/tmp \
    ${IMAGE_NAME}"}

# check if docker image exists on each node
check_image_on_node() {
    local node="$1"
    echo "Checking Docker image on node: $node"
    if srun --nodes=1 --ntasks=1 -w "$node" docker image ls --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
        echo "Docker image $IMAGE_NAME found on $node."
    else
        # load from local tar file
        echo "Docker image not found on $node, loading from tar file $IMAGE_FILE"
        srun --nodes=1 --ntasks=1 -w "$node" docker image load -i $IMAGE_FILE
    fi
}

for node in "${nodes_array[@]}"; do
    check_image_on_node "$node" &
done
wait
echo "Docker image is available on all nodes."

# dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun $DOCKER_RUN_CMD torchrun --nnodes 4 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ./torchtitan/train.py --job.config_file ${CONFIG_FILE}
# dcgmi profile --resume
