auto_scale_lr = dict(base_batch_size=32, enable=False)
base_lr = 0.004
classes = ("mouse",)
custom_hooks = [
    dict(
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        priority=49,
        type="EMAHook",
        update_buffers=True,
    ),
    dict(
        switch_epoch=140,
        switch_pipeline=[
            dict(type="PackDetInputs"),
        ],
        type="PipelineSwitchHook",
    ),
]
data_root = "/home/tc/datasets/mouse/"
dataset_type = "CocoDataset"
default_hooks = dict(
    checkpoint=dict(interval=50, save_best="coco/segm_mAP", type="CheckpointHook", max_keep_ckpts=1,),
    logger=dict(interval=5, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(draw=True, interval=20, type="DetVisualizationHook"),
)
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
file_client_args = dict(backend="disk")
launcher = "none"
load_from = (
    "./trained_models/2024-01-06_mouse_train_det/best_coco_segm_mAP_epoch_139.pth"
)
load_pipeline = [
    dict(file_client_args=dict(backend="disk"), type="LoadImageFromFile"),
    dict(poly2mask=True, type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.8,
            1.2,
        ),
        scale=(
            640,
            640,
        ),
        type="RandomResize",
    ),
    dict(
        allow_negative_crop=True,
        crop_size=(
            640,
            640,
        ),
        type="RandomCrop",
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        pad_val=dict(
            img=(
                114,
                114,
                114,
            )
        ),
        size=(
            640,
            640,
        ),
        type="Pad",
    ),
]
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint="torchvision://resnet50", type="Pretrained"),
        norm_cfg=dict(requires_grad=True, type="BN"),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style="pytorch",
        type="ResNet",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="DetDataPreprocessor",
    ),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type="FPN",
    ),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type="DeltaXYWHBBoxCoder",
            ),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type="L1Loss"),
            loss_cls=dict(loss_weight=1.0, type="CrossEntropyLoss", use_sigmoid=False),
            num_classes=80,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type="Shared2FCBBoxHead",
        ),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type="RoIAlign"),
            type="SingleRoIExtractor",
        ),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(loss_weight=1.0, type="CrossEntropyLoss", use_mask=True),
            num_classes=80,
            num_convs=4,
            type="FCNMaskHead",
        ),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type="RoIAlign"),
            type="SingleRoIExtractor",
        ),
        type="StandardRoIHead",
    ),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type="AnchorGenerator",
        ),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type="DeltaXYWHBBoxCoder",
        ),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type="L1Loss"),
        loss_cls=dict(loss_weight=1.0, type="CrossEntropyLoss", use_sigmoid=True),
        type="RPNHead",
    ),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type="nms"),
            score_thr=0.05,
        ),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type="nms"),
            nms_pre=1000,
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type="MaxIoUAssigner",
            ),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type="RandomSampler",
            ),
        ),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type="MaxIoUAssigner",
            ),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type="RandomSampler",
            ),
        ),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type="nms"),
            nms_pre=2000,
        ),
    ),
    type="MaskRCNN",
)
optim_wrapper = dict(
    loss_scale="dynamic",
    optimizer=dict(lr=0.004, type="AdamW", weight_decay=0.05),
    paramwise_cfg=dict(bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type="AmpOptimWrapper",
)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=1e-05, type="LinearLR"),
    dict(
        T_max=80,
        begin=80,
        by_epoch=True,
        convert_to_iter_based=True,
        end=160,
        eta_min=0.0002,
        type="CosineAnnealingLR",
    ),
]
resume = False
test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file="annotations/test.json",
        data_prefix=dict(img="images/test/"),
        data_root="./datasets/mouse/",
        metainfo=dict(classes=("mouse",)),
        pipeline=[
            dict(file_client_args=dict(backend="disk"), type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    640,
                    640,
                ),
                type="Resize",
            ),
            dict(
                pad_val=dict(
                    img=(
                        114,
                        114,
                        114,
                    )
                ),
                size=(
                    640,
                    640,
                ),
                type="Pad",
            ),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(
    ann_file="./datasets/mouse/annotations/test.json",
    format_only=False,
    metric=[
        "bbox",
        "segm",
    ],
    proposal_nums=(
        100,
        1,
        10,
    ),
    type="CocoMetric",
)
test_pipeline = [
    dict(file_client_args=dict(backend="disk"), type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scale=(
            640,
            640,
        ),
        type="Resize",
    ),
    dict(
        pad_val=dict(
            img=(
                114,
                114,
                114,
            )
        ),
        size=(
            640,
            640,
        ),
        type="Pad",
    ),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            140,
            1,
        ),
    ],
    max_epochs=160,
    type="EpochBasedTrainLoop",
    val_interval=1,
)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=32,
    dataset=dict(
        dataset=dict(
            ann_file="annotations/train.json",
            data_prefix=dict(img="images/train/"),
            data_root="./datasets/mouse/",
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(classes=("mouse",)),
            pipeline=[
                dict(file_client_args=dict(backend="disk"), type="LoadImageFromFile"),
                dict(
                    poly2mask=True,
                    type="LoadAnnotations",
                    with_bbox=True,
                    with_mask=True,
                ),
                dict(
                    keep_ratio=True,
                    ratio_range=(
                        0.8,
                        1.2,
                    ),
                    scale=(
                        640,
                        640,
                    ),
                    type="RandomResize",
                ),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        640,
                        640,
                    ),
                    type="RandomCrop",
                ),
                dict(type="YOLOXHSVRandomAug"),
                dict(prob=0.5, type="RandomFlip"),
                dict(
                    pad_val=dict(
                        img=(
                            114,
                            114,
                            114,
                        )
                    ),
                    size=(
                        640,
                        640,
                    ),
                    type="Pad",
                ),
            ],
            type="CocoDataset",
        ),
        max_refetch=60,
        pipeline=[
            dict(type="CopyPaste"),
            dict(
                allow_negative_crop=False,
                crop_size=(
                    640,
                    640,
                ),
                type="RandomCrop",
            ),
            dict(
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type="FilterAnnotations",
            ),
            dict(type="PackDetInputs"),
        ],
        type="MultiImageMixDataset",
    ),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_pipeline_stage2 = [
    dict(file_client_args=dict(backend="disk"), type="LoadImageFromFile"),
    dict(poly2mask=False, type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            640,
            640,
        ),
        type="RandomResize",
    ),
    dict(
        allow_negative_crop=True,
        crop_size=(
            640,
            640,
        ),
        recompute_bbox=True,
        type="RandomCrop",
    ),
    dict(
        min_gt_bbox_wh=(
            1,
            1,
        ),
        type="FilterAnnotations",
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        pad_val=dict(
            img=(
                114,
                114,
                114,
            )
        ),
        size=(
            640,
            640,
        ),
        type="Pad",
    ),
    dict(type="PackDetInputs"),
]
val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file="annotations/val.json",
        data_prefix=dict(img="images/val/"),
        data_root="./datasets/mouse/",
        metainfo=dict(classes=("mouse",)),
        pipeline=[
            dict(file_client_args=dict(backend="disk"), type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    640,
                    640,
                ),
                type="Resize",
            ),
            dict(
                pad_val=dict(
                    img=(
                        114,
                        114,
                        114,
                    )
                ),
                size=(
                    640,
                    640,
                ),
                type="Pad",
            ),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = dict(
    ann_file="./datasets/mouse/annotations/val.json",
    format_only=False,
    metric=[
        "bbox",
        "segm",
    ],
    proposal_nums=(
        100,
        1,
        10,
    ),
    type="CocoMetric",
)
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
work_dir = "./trained_models/2024-01-06_mouse_train_det"
