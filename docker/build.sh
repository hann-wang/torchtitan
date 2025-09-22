#!/bin/bash

IMAGE=ghcr.io/amd-agi/han-workspace:ubuntu22.04-pytorch2.10.0dev20250921-rocm6.4.2
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

docker build -t $IMAGE .
