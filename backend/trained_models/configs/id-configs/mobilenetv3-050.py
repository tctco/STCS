_base_ = ["./base.py"]
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="MobileNetV3",
        arch="small_050",
        norm_cfg=dict(type="BN", eps=1e-5, momentum=0.1),
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="StackedLinearClsHead",
        num_classes=1000,
        in_channels=288,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type="HSwish"),
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        init_cfg=dict(type="Normal", layer="Linear", mean=0.0, std=0.01, bias=0.0),
        topk=(1, 2),
    ),
)
