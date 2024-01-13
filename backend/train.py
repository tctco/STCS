from mmengine import Config
from mmengine.runner import Runner
import os.path as osp
import datetime
import torch

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("animal", help="animal name", type=str)
parser.add_argument(
    "--det", help="train detection model", action="store_true", default=False
)
parser.add_argument(
    "--pose", help="train pose model", action="store_true", default=False
)
parser.add_argument("--num_kpts", help="number of keypoints", type=int, default=0)
parser.add_argument(
    "--small", help="use small codec", action="store_true", default=False
)
parser.add_argument("--epochs", help="max epochs", default=160, type=int)
args = parser.parse_args()

# DET_CONFIG_PATH = (
#     f"/home/tc/open-mmlab/mmdetection/configs/mouse/rtmdet-ins_tiny_{args.animal}.py"
# )
# DET_CONFIG_PATH = '/home/tc/open-mmlab/mmdetection/configs/mouse/mask-rcnn_r50_fpn_1x_coco.py'
# DET_CONFIG_PATH = "./trained_models/mask-rcnn_r50_fpn.py"
DET_CONFIG_PATH = "./trained_models/ms-rcnn.py"
if args.small:
    POSE_CONFIG_PATH = (
        "/home/tc/open-mmlab/mmpose/configs/mouse/hm_mobilenetv2_small.py"
    )
else:
    POSE_CONFIG_PATH = "/home/tc/open-mmlab/mmpose/configs/mouse/hm_mobilenetv2.py"
# POSE_CONFIG_PATH = '/home/tc/open-mmlab/mmpose/projects/yolox-pose/configs/yolox-pose_tiny.py'
BATCH_SIZE = 6


def build_det_cfg(config_file_path, classes, dataset_root, max_epochs):
    cfg = Config.fromfile(config_file_path)
    cfg.train_cfg.max_epochs = max_epochs
    cfg.train_cfg.dynamic_intervals = [(max_epochs - 10, 1)]
    if "rcnn" not in config_file_path:
        cfg.custom_hooks[1]["switch_epoch"] = max_epochs - 10

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    work_dir = osp.join(f"./trained_models/{today}_{classes[0]}_det")
    cfg.work_dir = work_dir
    # cfg.train_dataloader.dataset.metainfo["classes"] = classes
    # cfg.val_dataloader.dataset.metainfo["classes"] = classes
    # cfg.test_dataloader.dataset.metainfo["classes"] = classes

    cfg.train_dataloader.dataset.data_root = dataset_root
    cfg.val_dataloader.dataset.data_root = dataset_root
    cfg.test_dataloader.dataset.data_root = dataset_root

    cfg.optim_wrapper.type = "AmpOptimWrapper"
    cfg.optim_wrapper.loss_scale = "dynamic"
    cfg.train_dataloader.batch_size = BATCH_SIZE
    cfg.val_dataloader.batch_size = BATCH_SIZE
    cfg.test_dataloader.batch_size = BATCH_SIZE
    cfg.val_evaluator.ann_file = osp.join(dataset_root, "annotations/val.json")
    cfg.test_evaluator.ann_file = osp.join(dataset_root, "annotations/test.json")
    cfg.auto_scale_lr.base_batch_size = BATCH_SIZE
    # cfg.optim_wrapper.optimizer.lr = 0.001
    # cfg.optim_wrapper.optimizer.weight_decay = 0.005
    return cfg


def build_pose_cfg(config_file_path, classes, dataset_root, num_kpts):
    cfg = Config.fromfile(config_file_path)
    if "rcnn" in config_file_path:
        cfg.model.roi_head.bbox_head.num_classes = len(classes)
        cfg.model.roi_head.mask_head.num_classes = len(classes)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    work_dir = osp.join(f"./trained_models/{today}_{classes[0]}_pose")
    cfg.work_dir = work_dir
    metainfo = (
        f"/home/tc/open-mmlab/mmpose/configs/_base_/datasets/coco_{classes[0]}.py"
    )
    cfg.train_dataloader.dataset.metainfo.from_file = metainfo
    cfg.val_dataloader.dataset.metainfo.from_file = metainfo
    cfg.test_dataloader.dataset.metainfo.from_file = metainfo

    cfg.train_dataloader.dataset.data_root = dataset_root
    cfg.val_dataloader.dataset.data_root = dataset_root
    cfg.test_dataloader.dataset.data_root = dataset_root
    cfg.train_dataloader.batch_size = BATCH_SIZE * 2
    cfg.auto_scale_lr.base_batch_size = BATCH_SIZE * 2

    cfg.val_evaluator.ann_file = osp.join(dataset_root, "annotations/val.json")
    cfg.test_evaluator.ann_file = osp.join(dataset_root, "annotations/test.json")
    if "yolo" in config_file_path:
        cfg.model.train_cfg.assigner.oks_calculator.metainfo = metainfo
        cfg.model.bbox_head.loss_pose.metainfo = metainfo
        cfg.model.bbox_head.head_module.num_keypoints = num_kpts
    else:
        cfg.model.head.out_channels = num_kpts
    return cfg


classes = (args.animal,)
dataset_root = f"/home/tc/datasets/{args.animal}/"
if args.det:
    torch.cuda.empty_cache()
    det_cfg = build_det_cfg(
        DET_CONFIG_PATH,
        classes=classes,
        dataset_root=dataset_root,
        max_epochs=args.epochs,
    )
    det_runner = Runner.from_cfg(det_cfg)
    det_runner.train()
if args.pose and args.num_kpts > 0:
    torch.cuda.empty_cache()
    pose_cfg = build_pose_cfg(
        POSE_CONFIG_PATH,
        classes=classes,
        dataset_root=dataset_root,
        num_kpts=args.num_kpts,
    )
    pose_runner = Runner.from_cfg(pose_cfg)
    pose_runner.train()
