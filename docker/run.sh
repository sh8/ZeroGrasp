#!/bin/bash

wandb docker-run -it \
  --mount type=bind,source="$(pwd)",target=/opt/app \
  --ipc=host \
  -v $HOME/.aws:/root/.aws \
  -v $HOME/.cache:/root/.cache \
  --gpus \"device=$1\" \
  zerograsp:latest /bin/zsh
