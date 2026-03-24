from __future__ import print_function, division
import sys
sys.path.append('core')

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from MTEV import MTEVStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder
from PIL import Image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))

        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
        # val = (valid_gt.flatten() >= 0.5)
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training',thing_test=True)
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)
    f = open('testkitti12.txt', 'a')
    f.write("Validation kitti: %f, %f\n" % (epe, d1))
    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}

# def canny_edge_mask(gray):
#     """生成边缘/非边缘掩膜"""
#     edges = cv2.Canny(gray, CANNY_THRESH1, CANNY_THRESH2)
#     # 可选：按梯度幅值阈值进一步筛选，这里简单二值化
#     mask_edge = edges.astype(bool)
#     # 确保非边缘与边缘不重叠
#     mask_non  = ~mask_edge
#     return mask_edge, mask_non

@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False, canny_low=50, canny_high=150):
    """ Perform validation using the Scene Flow (TEST) split
        and report metrics separately for edge / non-edge regions.
    """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    # 用于累积所有帧的误差
    epe_edge_all, epe_nedge_all = [], []
    d1_edge_all,  d1_nedge_all  = [], []

    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        # ---------- 原始前向 ----------
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)   # [2, H, W]
        assert flow_pr.shape == flow_gt.shape

        # ---------- 计算误差 ----------
        epe = torch.abs(flow_pr - flow_gt)                 # [2, H, W]
        epe = torch.norm(epe, dim=0)                       # [H, W]  每个像素的 EPE

        global_valid = (valid_gt >= 0.5) & (flow_gt.abs().max(dim=0)[0] < 192)  # [H, W]

        # ---------- Canny 边缘检测 ----------
        # 这里用 flow_gt 的幅值图做 Canny，你也可以换成 image1 的灰度图
        flow_mag_np = flow_gt.norm(dim=0).numpy()          # H, W
        flow_mag_np = (flow_mag_np / flow_mag_np.max() * 255).astype(np.uint8)
        edge_mask = cv2.Canny(flow_mag_np, canny_low, canny_high)   # H, W, uint8
        edge_mask = torch.from_numpy(edge_mask).bool()     # H, W

        # ---------- 分别计算两类掩膜下的指标 ----------
        valid_edge  = global_valid & edge_mask
        valid_nedge = global_valid & (~edge_mask)

        if valid_edge.sum() > 0:
            epe_edge_all.append(epe[valid_edge].mean().item())
            d1_edge_all.append((epe[valid_edge] > 1.0).float().mean().item())

        if valid_nedge.sum() > 0:
            epe_nedge_all.append(epe[valid_nedge].mean().item())
            d1_nedge_all.append((epe[valid_nedge] > 1.0).float().mean().item())

    # ---------- 汇总 ----------
    epe_edge  = np.mean(epe_edge_all)  if len(epe_edge_all)  else np.nan
    epe_nedge = np.mean(epe_nedge_all) if len(epe_nedge_all) else np.nan
    d1_edge   = 100 * np.mean(d1_edge_all)  if len(d1_edge_all)  else np.nan
    d1_nedge  = 100 * np.mean(d1_nedge_all) if len(d1_nedge_all) else np.nan

    with open('test.txt', 'a') as f:
        f.write(f"Validation Scene Flow (edge):  EPE={epe_edge:.3f}  D1={d1_edge:.2f}\n")
        f.write(f"Validation Scene Flow (n-edge): EPE={epe_nedge:.3f} D1={d1_nedge:.2f}\n")

    print(f"Validation Scene Flow (edge):  EPE={epe_edge:.3f}  D1={d1_edge:.2f}")
    print(f"Validation Scene Flow (n-edge): EPE={epe_nedge:.3f} D1={d1_nedge:.2f}")
    return {
        'scene-disp-epe-edge' : epe_edge,
        'scene-disp-d1-edge'  : d1_edge,
        'scene-disp-epe-nedge': epe_nedge,
        'scene-disp-d1-nedge' : d1_nedge,
    }


# CANNY_LOW   = 50
# CANNY_HIGH  = 150
# EDGE_DILATE = 1     # 可选：对边缘做膨胀，使边缘略粗
# # ---------------------------------

# @torch.no_grad()
# def validate_sceneflow(model, iters=32, mixed_prec=False):
#     """ Scene Flow TEST 集验证 + 边缘/非边缘分区统计 """
#     model.eval()
#     val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

#     # 全局统计量
#     epe_list, d1_list = [], []

#     # 边缘/非边缘统计量
#     epe_edge, epe_non = [], []
#     bad1_edge, bad1_non = [], []

#     for val_id in tqdm(range(len(val_dataset))):
#         _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

#         image1 = image1[None].cuda()
#         image2 = image2[None].cuda()

#         padder = InputPadder(image1.shape, divis_by=32)
#         image1, image2 = padder.pad(image1, image2)

#         with autocast(enabled=mixed_prec):
#             flow_pr = model(image1, image2, iters=iters, test_mode=True)
#         flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
#         assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
#         # 有效掩膜
#         valid = (valid_gt >= 0.5) & (flow_gt.abs() < 192)

#         # -------- 全局 --------
#         epe = torch.abs(flow_pr - flow_gt)
#         epe_list.append(epe[valid].mean().item())
#         d1_list.append((epe > 3.0)[valid].float().mean().item())

#         # -------- Canny 分区 --------
#         # 1. 把左图转灰度并做 Canny
#         # 1. 把左图也 pad 到 32 倍数，做 Canny
#         # left_pad = padder.pad(image1)[0]          # [C, Hp, Wp]
#         # left_np  = left_pad.squeeze(0).cpu().numpy()          # [C, Hp, Wp]
#         left_np = image1[:, :, :3] 
#         left_np  = (left_np.transpose(1, 2, 0) * 255).astype(np.uint8)
#         gray     = cv2.cvtColor(left_np, cv2.COLOR_RGB2GRAY)
#         edge_pad = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
#         if EDGE_DILATE:
#             kernel = np.ones((3, 3), np.uint8)
#             edge_pad = cv2.dilate(edge_pad, kernel, iterations=EDGE_DILATE)

#         # 2. 把掩膜 unpad 回原图尺寸
#         edge_mask = torch.from_numpy(edge_pad).bool()
#         edge_mask = padder.unpad(edge_mask.unsqueeze(0)).squeeze(0)  # [H, W]
#         non_mask  = ~edge_mask

#         # 2. 与有效视差区域取交集
#         edge_mask &= valid
#         non_mask  &= valid

#         if edge_mask.sum() > 0:
#             epe_edge.append(epe[edge_mask].mean().item())
#             bad1_edge.append((epe > 1.0)[edge_mask].float().mean().item())

#         if non_mask.sum() > 0:
#             epe_non.append(epe[non_mask].mean().item())
#             bad1_non.append((epe > 1.0)[non_mask].float().mean().item())

#     # ---------- 汇总 ----------
#     epe_all = np.mean(epe_list)
#     d1_all  = 100 * np.mean(d1_list)

#     epe_e   = np.mean(epe_edge)
#     bad1_e  = 100 * np.mean(bad1_edge)

#     epe_ne  = np.mean(epe_non)
#     bad1_ne = 100 * np.mean(bad1_non)

#     print(f"Validation Scene Flow:")
#     print(f"  EPE: {epe_all:.3f}, D1: {d1_all:.2f}%")
#     print(f"  Edge   - EPE: {epe_e:.3f}, >1px: {bad1_e:.2f}%")
#     print(f"  NonEdge- EPE: {epe_ne:.3f}, >1px: {bad1_ne:.2f}%")

#     with open('test.txt', 'a') as f:
#         f.write(f"SceneFlow EPE {epe_all:.3f} D1 {d1_all:.2f} "
#                 f"EdgeEPE {epe_e:.3f} Edge>1px {bad1_e:.2f} "
#                 f"NonEPE {epe_ne:.3f} Non>1px {bad1_ne:.2f}\n")

#     return {'scene-disp-epe': epe_all,
#             'scene-disp-d1':  d1_all,
#             'edge-epe':       epe_e,
#             'edge-bad1':      bad1_e,
#             'non-epe':        epe_ne,
#             'non-bad1':       bad1_ne}


@torch.no_grad()
def validate_usv(model, iters=32, mixed_prec=False):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    val_dataset = datasets.usvlanddatasets(dstype='Left_Img_Rectified')

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        # epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe = torch.abs(flow_pr - flow_gt)

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        if(np.isnan(epe[val].mean().item())):
            continue

        out = (epe > 3.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        # if val_id == 400:
        #     break

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    f = open('test.txt', 'a')
    f.write("Validation usvland: %f, %f\n" % (epe, d1))

    print("Validation usvland: %f, %f" % (epe, d1))
    return {'usvland-disp-epe': epe, 'usvland-disp-d1': d1}
@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < 192) & (occ_mask==255)
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()

    model = torch.nn.DataParallel(MTEVStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=args.mixed_precision)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

