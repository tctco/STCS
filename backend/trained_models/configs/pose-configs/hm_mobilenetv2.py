default_scope = "mmpose"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=50,
        save_best="coco/AP",
        rule="greater",
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=True),
)
custom_hooks = [dict(type="SyncBuffersHook")]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="PoseLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend")],
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True, num_digits=6)
log_level = "INFO"
load_from = None
resume = False
backend_args = dict(backend="local")
train_cfg = dict(by_epoch=True, max_epochs=160, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
optim_wrapper = dict(optimizer=dict(type="Adam", lr=0.0005))
param_scheduler = [
    dict(type="LinearLR", begin=0, end=15, start_factor=0.001, by_epoch=False),
    dict(
        type="MultiStepLR",
        begin=0,
        end=150,
        milestones=[50, 150],
        gamma=0.1,
        by_epoch=True,
    ),
]
auto_scale_lr = dict(base_batch_size=16)
codec = dict(type="MSRAHeatmap", input_size=(256, 256), heatmap_size=(64, 64), sigma=2)
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="MobileNetV2",
        widen_factor=1.0,
        out_indices=(7,),
        init_cfg=dict(type="Pretrained", checkpoint="mmcls://mobilenet_v2"),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=1280,
        out_channels=8,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=dict(
            type="MSRAHeatmap", input_size=(256, 256), heatmap_size=(64, 64), sigma=2
        ),
    ),
    test_cfg=dict(flip_test=True, flip_mode="heatmap", shift_heatmap=True),
)
dataset_type = "CocoDataset"
data_mode = "topdown"
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomFlip", direction="vertical"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=(256, 256)),
    dict(
        type="GenerateTarget",
        encoder=dict(
            type="MSRAHeatmap", input_size=(256, 256), heatmap_size=(64, 64), sigma=2
        ),
    ),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=(256, 256)),
    dict(type="PackPoseInputs"),
]
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CocoDataset",
        data_root="",
        data_mode="topdown",
        ann_file="train.json",
        data_prefix=dict(img=""),
        metainfo=dict(),
        pipeline=[
            dict(type="LoadImage"),
            dict(type="GetBBoxCenterScale"),
            dict(type="RandomFlip", direction="horizontal"),
            dict(type="RandomFlip", direction="vertical"),
            dict(type="RandomBBoxTransform"),
            dict(type="TopdownAffine", input_size=(256, 256)),
            dict(
                type="GenerateTarget",
                encoder=dict(
                    type="MSRAHeatmap",
                    input_size=(256, 256),
                    heatmap_size=(64, 64),
                    sigma=2,
                ),
            ),
            dict(type="PackPoseInputs"),
        ],
    ),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="CocoDataset",
        data_root="",
        data_mode="topdown",
        ann_file="val.json",
        data_prefix=dict(img=""),
        metainfo=dict(),
        test_mode=True,
        pipeline=[
            dict(type="LoadImage"),
            dict(type="GetBBoxCenterScale"),
            dict(type="TopdownAffine", input_size=(256, 256)),
            dict(type="PackPoseInputs"),
        ],
    ),
)
test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="CocoDataset",
        data_root="",
        data_mode="topdown",
        ann_file="test.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=[
            dict(type="LoadImage"),
            dict(type="GetBBoxCenterScale"),
            dict(type="TopdownAffine", input_size=(256, 256)),
            dict(type="PackPoseInputs"),
        ],
    ),
)
val_evaluator = dict(type="CocoMetric", ann_file="val.json")
test_evaluator = dict(type="CocoMetric", ann_file="test.json")
work_dir = ""
