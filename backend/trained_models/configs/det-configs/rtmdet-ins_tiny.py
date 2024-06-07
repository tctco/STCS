default_scope = "mmdet"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=5),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=50,
        save_best="coco/segm_mAP",
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook", draw=True, interval=20),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")],
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=160,
    val_interval=1,
    dynamic_intervals=[(140, 1)],
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-05, by_epoch=False, begin=0, end=1000),
    dict(
        type="CosineAnnealingLR",
        eta_min=0.0002,
        begin=75,
        end=160,
        T_max=75,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    loss_scale="dynamic",
)
auto_scale_lr = dict(enable=False, base_batch_size=8)
dataset_type = "CocoDataset"
data_root = "/home/tc/datasets/mouse/"
classes = ("mouse",)
file_client_args = dict(backend="disk")

train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type="MultiImageMixDataset",
        max_refetch=60,
        dataset=dict(
            type="CocoDataset",
            ann_file="annotations/train.json",
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[
                dict(type="LoadImageFromFile", file_client_args=dict(backend="disk")),
                dict(
                    type="LoadAnnotations",
                    with_bbox=True,
                    with_mask=True,
                    poly2mask=True,
                ),
                dict(
                    type="RandomResize",
                    scale=(640, 640),
                    ratio_range=(0.8, 1.2),
                    keep_ratio=True,
                ),
                dict(type="RandomCrop", crop_size=(640, 640), allow_negative_crop=True),
                dict(type="YOLOXHSVRandomAug"),
                dict(type="RandomFlip", prob=0.5),
                dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
            ],
        ),
        pipeline=[
            dict(type="CopyPaste"),
            dict(type="RandomCrop", crop_size=(640, 640), allow_negative_crop=False),
            dict(
                type="CachedMixUp",
                img_scale=(640, 640),
                ratio_range=(1.0, 1.0),
                max_cached_images=10,
                random_pop=False,
                pad_val=(114, 114, 114),
                prob=0.2,
            ),
            dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
            dict(type="PackDetInputs"),
        ],
    ),
    pin_memory=True,
)
val_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        metainfo=dict(classes=("mouse",)),
        type="CocoDataset",
        data_root="/home/tc/datasets/mouse/",
        ann_file="annotations/val.json",
        data_prefix=dict(img="images/val/"),
        test_mode=True,
        pipeline=[
            dict(type="LoadImageFromFile", file_client_args=dict(backend="disk")),
            dict(type="Resize", scale=(640, 640), keep_ratio=True),
            dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
            dict(
                type="PackDetInputs",
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
            ),
        ],
    ),
)
test_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        metainfo=dict(classes=("mouse",)),
        type="CocoDataset",
        data_root="/home/tc/datasets/mouse/",
        ann_file="annotations/test.json",
        data_prefix=dict(img="images/test/"),
        test_mode=True,
        pipeline=[
            dict(type="LoadImageFromFile", file_client_args=dict(backend="disk")),
            dict(type="Resize", scale=(640, 640), keep_ratio=True),
            dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
            dict(
                type="PackDetInputs",
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
            ),
        ],
    ),
)
val_evaluator = dict(
    type="CocoMetric",
    ann_file="/home/tc/datasets/mouse/annotations/val.json",
    metric=["bbox", "segm"],
    format_only=False,
    proposal_nums=(100, 1, 10),
)
test_evaluator = dict(
    type="CocoMetric",
    ann_file="/home/tc/datasets/mouse/annotations/test.json",
    metric=["bbox", "segm"],
    format_only=False,
    proposal_nums=(100, 1, 10),
)
model = dict(
    type="RTMDet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU", inplace=True),
        init_cfg=dict(
            type="Pretrained",
            prefix="backbone.",
            checkpoint="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth",
        ),
    ),
    neck=dict(
        type="CSPNeXtPAFPN",
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    bbox_head=dict(
        type="RTMDetInsSepBNHead",
        num_classes=1,
        in_channels=96,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=96,
        act_cfg=dict(type="SiLU", inplace=True),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        anchor_generator=dict(type="MlvlPointGenerator", offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type="DistancePointBBoxCoder"),
        loss_cls=dict(
            type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_mask=dict(type="DiceLoss", loss_weight=2.0, eps=5e-06, reduction="mean"),
    ),
    train_cfg=dict(
        assigner=dict(type="DynamicSoftLabelAssigner", topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.5,
        nms=dict(type="soft_nms", iou_threshold=0.5),
        max_per_img=100,
        mask_thr_binary=0.5,
    ),
)
max_epochs = 300
stage2_num_epochs = 20
interval = 10
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="PipelineSwitchHook",
        switch_epoch=140,
        switch_pipeline=[dict(type="PackDetInputs")],
    ),
]
work_dir = "./trained_models/2023-07-12_mouse_det"
