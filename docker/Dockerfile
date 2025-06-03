# ===
# Dockerfile to train and eval ZeroGrasp
#   => Base Image :: Python 3.11, Pytorch 2.2, CUDA 12.1.0
# ===

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

RUN apt-get update
RUN apt-get install -y software-properties-common

RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
      tar xz-utils cmake curl docker.io ffmpeg git htop zsh tmux && \
    apt-get autoclean -y && apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz /opt/
RUN tar -xf /opt/s5cmd_2.2.2_Linux-64bit.tar.gz -C /opt/ && \
    mv /opt/s5cmd /usr/bin/ && rm /opt/s5cmd_2.2.2_Linux-64bit.tar.gz

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
ENV ZSH_THEME="agnoster"

RUN pip install ninja
RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
RUN pip install -U xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121

ENV TORCH_CUDA_ARCH_LIST="8.0+PTX"

COPY requirements.txt /opt/app/
WORKDIR /opt/app
RUN pip install -r requirements.txt

RUN git config --global --add safe.directory /opt/app

COPY submodules/octree_feature_extractor/ /opt/app/submodules/octree_feature_extractor
WORKDIR /opt/app/submodules/octree_feature_extractor/
RUN python setup.py install

RUN pip install graspnetAPI --no-deps
RUN pip install transforms3d autolab_core cvxopt grasp_nms
RUN pip install torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
RUN apt-get update && apt-get install -y vim

WORKDIR /opt/app
