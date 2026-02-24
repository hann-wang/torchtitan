#!/bin/bash

IMAGE=ghcr.io/amd-agi/han-workspace:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

docker build -t $IMAGE .
docker push $IMAGE
