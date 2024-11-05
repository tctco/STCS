# defaults to use registries in mmcls
default_scope = "mmpretrain"

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type="IterTimerHook"),
    # print log every 100 iterations.
    logger=dict(type="LoggerHook", interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type="ParamSchedulerHook"),
    # save checkpoint per epoch.
    checkpoint=dict(type="CheckpointHook", interval=1),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type="DistSamplerSeedHook"),
    # validation results visualization, set True to enable it.
    visualization=dict(type="VisualizationHook", enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# dataset settings
dataset_type = "BaseDataset"
data_root = "/home/tc/open-mmlab/tracker/output/cropped/"
data_preprocessor = dict(
    num_classes=4,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RandomCrop",
        crop_size=(192, 96),
        pad_if_needed=True,
        pad_val=(255, 255, 255),
    ),
    # dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="Rotate", angle=180, prob=0.05),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RandomCrop",
        crop_size=(192, 96),
        pad_if_needed=True,
        pad_val=(255, 255, 255),
    ),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=196,
    num_workers=2,
    dataset=dict(
        type="ClassBalancedDataset",
        oversample_thr=0.25,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file="train.json",
            data_prefix="",
            pipeline=train_pipeline,
        ),
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=196,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="val.json",
        data_prefix="",
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
# val_evaluator = dict(type="Accuracy", topk=(1, 2))
val_evaluator = [
    dict(type="AveragePrecision"),
    dict(type="Accuracy", topk=(1, 2)),
    dict(type="SingleLabelMetric"),
]
# val_evaluator = dict(type="AveragePrecision")

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    optimizer=dict(type="Adam", lr=0.001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
)  # weight_decay极为有害


# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)
val_cfg = dict()
test_cfg = dict()

resume = False
load_from = "./osnet_x0_25_imagenet_renamed.pth"

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=196)

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(
        type="CheckpointHook", interval=30, save_best="auto", max_keep_ckpts=1
    ),
    visualization=dict(type="VisualizationHook", enable=True),
    logger=dict(type="LoggerHook", interval=100),
)

# https://mmengine.readthedocs.io/zh_CN/latest/api/generated/mmengine.hooks.EarlyStoppingHook.html#mmengine.hooks.EarlyStoppingHook
custom_hooks = [
    dict(
        type="mmengine.EarlyStoppingHook",
        monitor="accuracy/top1",
        min_delta=0.5,
        patience=5,
        stopping_threshold=99.5,
    )
]
