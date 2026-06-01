#!/bin/bash

CONTAINER_NAME=ubuntu22_cuda128_${USER}_rlinf
IMAGE_BASE=rlinf:ssh


docker run -d -t \
  --cap-add=SYS_PTRACE \
  --gpus all \
  --shm-size 100g \
  --name ${CONTAINER_NAME} \
  --privileged \
  --net=host \
  --mount type=bind,source=$HOME/workspace,target=/mnt/workspace \
  --mount type=bind,source=/mnt/hdd2/tung/workspace/,target=/mnt/workspace2 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  ${IMAGE_BASE}