# ------------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------
# Modified by Fei Zhang
# ------------------------------------------------------------------------------
import argparse
import os
import os.path as osp
import mmcv
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datasets import build_text_transform
from main_pg_seg import validate_seg
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import set_random_seed
from models import build_model
from omegaconf import OmegaConf, read_write
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
import torch.nn.functional as F
BUILD_SEG_DATALOADER = build_seg_dataloader
from utils import get_config, get_logger, load_checkpoint
import numpy as np
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def parse_args():
    parser = argparse.ArgumentParser('PGSeg segmentation evaluation and visualization')
    parser.add_argument(
        '--cfg',
        type=str,
        # default='./configs/group_vit_gcc_redcap_30e.yml',
        required=True,
        help='path to config file',
    )
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument(
        '--output', type=str, help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>',
        default = 'vis_img'
        )
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument(
        '--vis',
        help='Specify the visualization mode, '
        'could be a list, support [input, pred, input_pred, all_groups, first_group, final_group, input_pred_label] ',
        default=['input','input_pred','final_group','input_pred_label'],
        nargs='+')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=True, help='local rank for DistributedDataParallel')


    args = parser.parse_args()

    return args


def inference(cfg):
    logger = get_logger()


    data_loader = BUILD_SEG_DATALOADER(build_seg_dataset(cfg.evaluate.seg))
    dataset = data_loader.dataset

    logger.info(f'Evaluating dataset: {dataset}')

    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    model.cuda()
    logger.info(str(model))

    if cfg.train.amp_opt_level != 'O0':
        model = amp.initialize(model, None, opt_level=cfg.train.amp_opt_level)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    load_checkpoint(cfg, model, None, None)

    if 'seg' in cfg.evaluate.task:
        miou = validate_seg(cfg, data_loader, model)
        logger.info(f'mIoU of the network on the {len(data_loader.dataset)} test images: {miou:.2f}%')
    else:
        logger.info('No segmentation evaluation specified')

    # if cfg.vis:
    #     vis_seg(cfg, data_loader, model, cfg.vis)


@torch.no_grad()
def vis_seg(config, data_loader, model, vis_modes):
    dist.barrier()
    model.eval()

    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    if config.evaluate.seg['cfg'].split('/')[-1] == 'imagenet_s.py':
        imgnet_flag = True
    else:
        imgnet_flag = False
    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)
    seg_model = build_seg_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.seg)

    mmddp_model = MMDistributedDataParallel(
        seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    mmddp_model.eval()
    model = mmddp_model.module
    device = next(model.parameters()).device
    dataset = data_loader.dataset

    if dist.get_rank() == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            # Here 'rescale=False' for ImageNet-S, otherwise CUDA would trigger out-of-memory
            result = mmddp_model(return_loss=False, rescale=not imgnet_flag,**data)

        # (True Label Guidance), introduce the true label for
        if config.evaluate.seg['true_label_guidance']:
            seg_map = dataset.get_gt_seg_map_by_idx(index=batch_indices[0])
            seg_map_cls = np.unique(seg_map)
            seg_map_cls = np.delete(seg_map_cls,np.where(seg_map_cls==dataset.ignore_index))
            seg_map_cls = torch.from_numpy(seg_map_cls).long()
            seg_map_cls_mask = torch.zeros((result[0].shape[0],1,1))
            seg_map_cls_mask[seg_map_cls] = 1
            pred = torch.argmax(result[0]*seg_map_cls_mask.cuda(), dim=0)
        else:
            pred = torch.argmax(result[0],dim=0)  #*seg_map_cls_mask.cuda() if s
        result = [pred.cpu().numpy()]

        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for batch_idx, img, img_meta in zip(batch_indices, imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            if not imgnet_flag:
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h)) #Comment when meeting imagenet
            for vis_mode in vis_modes:
                out_file = osp.join(config.output, 'vis_imgs', vis_mode, img_meta['ori_filename'])
                model.show_result(img_show, img_tensor.to(device), result, out_file, vis_mode)
            if dist.get_rank() == 0:
                batch_size = len(result) * dist.get_world_size()
                for _ in range(batch_size):
                    prog_bar.update()


def main():
    args = parse_args()
    cfg = get_config(args)

    if cfg.train.amp_opt_level != 'O0':
        assert amp is not None, 'amp not installed!'

    with read_write(cfg):
        cfg.evaluate.eval_only = True


    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(cfg.local_rank)

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    dist.barrier()


    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config saved to {path}')

    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    inference(cfg)
    dist.barrier()


if __name__ == '__main__':
    main()
