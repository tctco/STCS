# DET_CONFIG_FILE = "./trained_models/2023-07-12_mouse_det/rtmdet-ins_tiny.py"
# DET_CHECKPOINT_FILE = (
#     "./trained_models/2023-07-12_mouse_det/best_coco_segm_mAP_epoch_155.pth"
# )

# DET_CONFIG_FILE = "./trained_models/2024-01-04_mouse_det/ms-rcnn.py"
# DET_CHECKPOINT_FILE = (
#     "./trained_models/2024-01-04_mouse_det/best_coco_segm_mAP_epoch_91.pth"
# )

DET_CHECKPOINT_FILE = "./trained_models/yolov8seg-s/weights/best.pt"
DET_CONFIG_FILE = "YOLO"

# POSE_CONFIG_FILE = "/home/tc/open-mmlab/mmpose/configs/mouse/hm_mobilenetv2.py"
# POSE_CHECKPOINT_FILE = (
#     "./trained_models/2023-06-10_mouse_pose/best_coco_AP_epoch_86.pth"
# )

POSE_CONFIG_FILE = "YOLO"
POSE_CHECKPOINT_FILE = "./trained_models/yolov8pose/weights/best.pt"


# DET_CONFIG_FILE = "./trained_models/2023-04-23_ant_det/rtmdet-ins_tiny.py"
# DET_CHECKPOINT_FILE = (
#     "./trained_models/2023-04-23_ant_det/best_coco/segm_mAP_epoch_137.pth"
# )
# POSE_CONFIG_FILE = "./trained_models/2023-06-03_ant_pose/config.py"
# POSE_CHECKPOINT_FILE = "./trained_models/2023-06-03_ant_pose/best_coco_AP_epoch_159.pth"

# DET_CONFIG_FILE = "./trained_models/2023-06-01_fly_det/config.py"
# DET_CHECKPOINT_FILE = (
#     "./trained_models/2023-06-01_fly_det/best_coco_segm_mAP_epoch_128.pth"
# )
# POSE_CONFIG_FILE = "./trained_models/2023-06-03_fly_pose/config.py"
# POSE_CHECKPOINT_FILE = "./trained_models/2023-06-03_fly_pose/best_coco_AP_epoch_139.pth"


# CLS_CONFIG_FILE = "./trained_models/osnet/osnet_x0_25.py"
CLS_CONFIG_FILE = "./trained_models/arcface/osnet_x0_25.py"
CLS_CHECKPOINT_FILE = "./trained_models/arcface/osnet_x0_25_imagenet_renamed.pth"
VISUALIZE_MASK = False
VIDEO_NAME = "cropped_cage3r1r2"
MAX_DET = 4
VIDEO_PATH = f"/mnt/e/projects/videos/{VIDEO_NAME}.mp4"
CONF_THRESH = 0.5
# MIN_CLS_CONF = 0.65
MIN_CLS_CONF = 0.6
NMS_THRESH = 0.5
DRAW_HEATMAP = False
IOU_THRESHOLD = 0.7
UPPER_KEYPOINT = 0
LOWER_KEYPOINT = 5
# IOU_THRESHOLD = 30
# IOU_LOWER_THRESHOLD = 60
NUM_KPTS = 8
NUM_LINKS = 7
# PAD_VALUE = [104, 116, 124]
DET_MODEL = "RTMDET"
# DET_MODEL = 'MS_RCNN'
DET_BATCH_SIZE = 8
BATCH_SIZE = 256
MIN_MERGE_FRAMES = 150
NUM_KPTS = 8
MAX_STRIDE = 15
MIN_STRIDE = 2
INFERENCE_STRIDE = 3
ACCUMULATED_FRAME_RATIO = 0.9
DISTANCE_TYPE = "cosine"
APPEARANCE_THRESHOLD = 0.05
RADIUS = 4  # kpt radius
MAX_AGE = 30
SOFT_BORDER = MAX_AGE
MAX_CROP_SIZE = 256
MAX_WIDTH_HEIGHT = 640  # set to -1 to disable

ENABLE_FLOW = False

FLOW_CHECKPOINT_FILE = (
    "./trained_models/liteflownet2/liteflownet2_pre_M3S3R3_8x1_flyingchairs_320x448.pth"
)
FLOW_CONFIG_FILE = (
    "./trained_models/liteflownet2/liteflownet2-pre-M3S3R3_8xb1_flyingchairs-320x448.py"
)

# FLOW_CHECKPOINT_FILE = ""
# FLOW_CONFIG_FILE = "opencv"

ANIMAL_CONFIGS = {
    "mouse": {
        "connections": [[0, 1], [0, 2], [0, 5], [5, 3], [5, 4], [5, 6], [6, 7]],
        "nkpts": 8,
    },
    "ant": {
        "connections": [[0, 1], [1, 2], [0, 3], [3, 4], [0, 5], [5, 6]],
        "nkpts": 7,
    },
    "fly": {
        "connections": [[2, 0], [2, 1], [0, 1], [2, 3], [3, 4], [3, 5], [2, 4], [2, 5]],
        "nkpts": 6,
    },
}

MODELS_CONFIGS = {
    "2023-04-23_ant_det": {
        "config": "./trained_models/2023-04-23_ant_det/rtmdet-ins_tiny.py",
        "checkpoint": "./trained_models/2023-04-23_ant_det/best_coco/segm_mAP_epoch_137.pth",
        "animal": "ant",
        "type": "segm",
        "method": "RTMDet",
    },
    "2023-04-23_ant_pose": {
        "config": "./trained_models/2023-04-23_ant_pose/hm_mobilenetv2.py",
        "checkpoint": "./trained_models/2023-04-23_ant_pose/best_coco/AP_epoch_114.pth",
        "animal": "ant",
        "type": "pose",
        "method": "MobileNetV2+SimpleBaseline",
    },
    "2023-06-01_fly_det": {
        "config": "./trained_models/2023-06-01_fly_det/config.py",
        "checkpoint": "./trained_models/2023-06-01_fly_det/best_coco_segm_mAP_epoch_128.pth",
        "animal": "fly",
        "type": "segm",
        "method": "RTMDet",
    },
    "2023-06-03_ant_pose": {
        "config": "./trained_models/2023-06-03_ant_pose/config.py",
        "checkpoint": "./trained_models/2023-06-03_ant_pose/best_coco_AP_epoch_159.pth",
        "animal": "ant",
        "type": "pose",
        "method": "MobileNetV2+SimpleBaseline",
    },
    "2023-06-03_fly_pose": {
        "config": "./trained_models/2023-06-03_fly_pose/config.py",
        "checkpoint": "./trained_models/2023-06-03_fly_pose/best_coco_AP_epoch_139.pth",
        "animal": "fly",
        "type": "pose",
        "method": "MobileNetV2+SimpleBaseline",
    },
    "2023-06-10_mouse_pose": {
        "config": "./trained_models/2023-06-10_mouse_pose/hm_mobilenetv2.py",
        "checkpoint": "./trained_models/2023-06-10_mouse_pose/best_coco_AP_epoch_86.pth",
        "animal": "mouse",
        "type": "pose",
        "method": "MobileNetV2+SimpleBaseline",
    },
    "2023-07-12_mouse_det": {
        "config": "./trained_models/2023-07-12_mouse_det/rtmdet-ins_tiny.py",
        "checkpoint": "./trained_models/2023-07-12_mouse_det/best_coco_segm_mAP_epoch_155.pth",
        "animal": "mouse",
        "type": "segm",
        "method": "RTMDet",
    },
    "2024-01-06_mouse_train_det": {
        "config": "./trained_models/2024-01-06_mouse_train_det/ms-rcnn.py",
        "checkpoint": "./trained_models/2024-01-06_mouse_train_det/best_coco_segm_mAP_epoch_139.pth",
        "animal": "mouse",
        "type": "segm",
        "method": "MS RCNN",
    },
    "yolov8pose": {
        "config": "YOLO",
        "checkpoint": "./trained_models/yolov8pose/weights/best.pt",
        "animal": "mouse",
        "type": "pose",
        "method": "YOLOv8 Pose",
    },
    "yolov8seg-s": {
        "config": "YOLO",
        "checkpoint": "./trained_models/yolov8seg-s/weights/best.pt",
        "animal": "mouse",
        "type": "segm",
        "method": "YOLOv8 YOLACT",
    },
    "liteflownet2": {
        "config": "./trained_models/liteflownet2/liteflownet2-pre-M3S3R3_8xb1_flyingchairs-320x448.py",
        "checkpoint": "./trained_models/liteflownet2/liteflownet2_pre_M3S3R3_8x1_flyingchairs_320x448.pth",
        "animal": "",
        "type": "flow",
        "method": "LiteFlowNet2",
    },
    "Farneback": {
        "config": "opencv",
        "checkpoint": "",
        "animal": "",
        "type": "flow",
        "method": "Farneback",
    },
}
