#!/bin/bash

IMAGE=ghcr.io/amd-agi/han-workspace:ubuntu24.04-pytorch2.10.0dev20250922-rocm7.0
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

docker build -t $IMAGE -f Dockerfile.mi355 .
