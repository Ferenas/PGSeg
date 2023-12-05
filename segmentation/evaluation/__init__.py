# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------
# Modified by Fei Zhang
# -------------------------------------------------------------------------

from .builder import build_seg_dataloader, build_seg_dataset, build_seg_demo_pipeline, build_seg_inference
from .pgseg_seg import PGSegInference, GROUP_PALETTE
__all__ = [
     'build_seg_dataset', 'build_seg_dataloader', 'build_seg_inference',
    'build_seg_demo_pipeline', 'GROUP_PALETTE', 'PGSegInference'
]
