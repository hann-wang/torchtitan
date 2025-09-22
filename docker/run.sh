#!/bin/bash

IMAGE=ghcr.io/amd-agi/han-workspace:ubuntu22.04-pytorch2.10.0dev20250921-rocm6.4.2
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
GID_RENDER=$(getent group render | cut -d: -f3)
GID_VIDEO=$(getent group video | cut -d: -f3)

docker run --rm -it \
    --ulimit core=0  --privileged \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add $GID_RENDER \
    --group-add $GID_VIDEO \
    --ipc=host --shm-size 8G \
    --workdir /workspace \
    -v $SCRIPT_DIR/../..:/workspace \
    $IMAGE
