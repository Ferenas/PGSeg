#!/usr/bin/env bash

# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------


SCRIPT=$1  #The corresponding command index in the terminal $1
CONFIG=$2  #$2  --master_port=22223 -m torch.distributed.launch --nproc_per_node=$GPUS
GPUS=$3  #$3
PORT=${PORT:20000}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=$GPUS\
    $SCRIPT --cfg $CONFIG ${@:4}
