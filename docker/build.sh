#!/bin/bash

IMAGE=wanghanthu/workspace:ubuntu22.04-pytorch2.14.0dev20260707-rocm7.2
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

docker build -t $IMAGE .
docker push $IMAGE
