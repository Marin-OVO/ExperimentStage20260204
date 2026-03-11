import os
import csv
import pandas as pd

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from model import UNetTestAtt
from datasets import CrowdDataset
from utils import custom_collate_fn
from utils.metrics import PointsMetrics
from utils.logger import setup_default_logging, time_str
from utils.seed import set_seed
from utils.lmds import LMDS
from model.utils import load_model

import albumentations as A
from datasets.transforms import DownSample


def vis(args):

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_path, exist_ok=True)

    # output with time
    timestr = time_str()
    work_dir = os.path.join(args.output_path, f'vis_{timestr}')
    os.makedirs(work_dir, exist_ok=True)

    vis_dir = os.path.join(work_dir, 'vis')
    tf_dir = os.path.join(work_dir, 'vis_tf')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(tf_dir, exist_ok=True)

    logger_path = os.path.join(work_dir, 'log')
    os.makedirs(logger_path, exist_ok=True)

    logger, timestr = setup_default_logging('vis', logger_path)

    logger.info('=' * 60)
    logger.info('Visualization Configuration:')
    for arg, value in vars(args).items():
        logger.info(f'{arg}: {value}')

    model = UNetTestAtt(in_channels=3, num_class=args.num_classes, bilinear=args.bilinear)
    model = load_model(model, args.checkpoint_path, strict=False)
    model.to(device)
    model.eval()

    logger.info(f'Model loaded from {args.checkpoint_path}')

    # dataset
    val_albu_transforms = [
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ]
    val_end_transforms = [
        DownSample(down_ratio=args.ds_down_ratio,
                   crowd_type=args.ds_crowd_type)
    ]

    val_dataset = CrowdDataset(
        data_root=args.data_root,
        train=False,
        train_list="crowd_train.list",
        val_list="crowd_val.list",
        albu_transforms=val_albu_transforms,
        end_transforms=val_end_transforms
    )

    logger.info(f'Val dataset size: {len(val_dataset)}')

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_worker,
    )

    ks = args.lmds_kernel_size
    lmds = LMDS(kernel_size=(ks, ks), adapt_ts=args.lmds_adapt_ts)

    metrics = PointsMetrics(radius=args.radius, num_classes=args.num_classes)

    def draw_points(img, points, drawer, cfg):
        for p in points:
            x, y = int(p[0]), int(p[1])
            drawer(img, (x, y), **cfg)

    pred_draw_cfg = (
        cv2.circle,
        dict(radius=4, color=(0, 0, 255), thickness=-1)  # 红色预测点
    )

    draw_cfg = [
        ("tp", cv2.circle, dict(
            radius=4,
            color=(255, 255, 0),
            thickness=-1
        )),
        ("fp", cv2.circle, dict(
            radius=4,
            color=(255, 0, 255),
            thickness=2
        )),
        ("fn", cv2.drawMarker, dict(
            color=(0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=8,
            thickness=2
        ))
    ]

    logger.info('=' * 60)
    logger.info('Start Visualization:')

    metrics.flush()
    with torch.no_grad():
        for idx, (image, target) in enumerate(val_dataloader):

            image = image.to(device)
            img_path = target['img_path'][0]

            raw_img = cv2.imread(img_path)
            H0, W0 = raw_img.shape[:2]

            gt_points = target['points'][0].cpu().numpy()  # (x, y)
            gt_loc = [(float(x), float(y)) for x, y in gt_points]
            gt = dict(
                loc=gt_loc,
                labels=[1] * len(gt_loc)
            )

            outputs = model(image)

            counts, locs, labels, scores = lmds(outputs['heatmap_out'])
            locs_list = locs[0]
            labels_list = labels[0]
            scores_list = scores[0]

            down_ratio = args.ds_down_ratio
            pred_loc = [
                (float(x) * down_ratio, float(y) * down_ratio)
                for (y, x), label in zip(locs_list, labels_list)
                if label == 1  # only fg
            ]
            pred = dict(
                loc=pred_loc,
                labels=[1] * len(pred_loc),
                scores=[s for (y, x), label, s in zip(locs_list, labels_list, scores_list) if label == 1]
            )

            # pred vis
            img_pred = raw_img.copy()
            drawer, cfg = pred_draw_cfg
            draw_points(img_pred, pred['loc'], drawer, cfg)
            cv2.imwrite(
                os.path.join(vis_dir, os.path.basename(img_path)),
                img_pred
            )

            # tp/fp/fn vis
            metrics.feed(gt=gt, preds=pred)

            tp = metrics.current_tp if metrics.current_tp else []
            fp = metrics.current_fp if metrics.current_fp else []
            fn = metrics.current_fn if metrics.current_fn else []

            img_tf = raw_img.copy()
            for name, drawer, cfg in draw_cfg:
                pts = dict(tp=tp, fp=fp, fn=fn)[name]
                draw_points(img_tf, pts, drawer, cfg)
            cv2.imwrite(
                os.path.join(tf_dir, os.path.basename(img_path)),
                img_tf
            )

            logger.info(f"[{idx + 1:3d}/{len(val_dataset)}] "
                        f"GT: {len(gt_loc):4d} | Pred: {len(pred_loc):4d} | "
                        f"saved {os.path.basename(img_path)}")

    logger.info('=' * 60)

    mAP = np.mean([metrics.ap(c) for c in range(1, metrics.num_classes)]).item()
    metrics.aggregate()

    recall    = metrics.recall()
    precision = metrics.precision()
    f1_score  = metrics.fbeta_score()

    logger.info(
        f"Test Results: "
        f"Precision: {precision:^8.4f} | "
        f"Recall: {recall:^8.4f} | "
        f"F1-score: {f1_score:^8.4f} | "
        f"mAP: {mAP:^8.4f}"
    )

    # save to csv
    csv_path = os.path.join(work_dir, 'test_metrics.csv')
    results = {
        'Precision': precision,
        'Recall':    recall,
        'F1-score':  f1_score,
        'mAP':       mAP,
    }
    data_frame = pd.DataFrame([results])
    data_frame.to_csv(csv_path, index=False)

    logger.info(f'Test metrics saved to {csv_path}')
    logger.info(f'Vis images saved to  {vis_dir}')
    logger.info(f'TF  images saved to  {tf_dir}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch Visualization')

    parser.add_argument('--data_root',        default='data/crowdsat',          type=str)
    parser.add_argument('--checkpoint_path',  default='weights/best_model.pth', type=str)
    parser.add_argument('--output_path',      default='vis',                    type=str)
    parser.add_argument('--num_classes',      default=2,      type=int)
    parser.add_argument('--bilinear',         default=True,   type=bool)
    parser.add_argument('--device',           default='cuda', type=str)
    parser.add_argument('--num_worker',       default=0,       type=int)
    parser.add_argument('--ds_down_ratio',    default=1,       type=int)
    parser.add_argument('--ds_crowd_type',    default='point', type=str)
    parser.add_argument('--lmds_kernel_size', default=3,   type=int)
    parser.add_argument('--lmds_adapt_ts',    default=0.5, type=float)
    parser.add_argument('--radius',           default=2,    type=int)
    parser.add_argument('--seed',             default=42,   type=int)

    args = parser.parse_args()
    vis(args)