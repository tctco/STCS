from dataclasses import dataclass
from typing import Optional
import os
import random
from redis import Redis
from rq import Queue
import subprocess
from sqlalchemy.orm import sessionmaker, joinedload
import numpy as np
from ast import literal_eval
from pathlib import Path
import json
import os
import tempfile
from app.common.database import engine
from constants import (
    NMS_THRESH,
    CONF_THRESH,
    MAX_AGE,
    DET_BATCH_SIZE,
    IOU_THRESHOLD,
    MAX_WIDTH_HEIGHT,
    CLS_CONFIG_FILE,
    CLS_CHECKPOINT_FILE,
    BATCH_SIZE,
    SOFT_BORDER,
    ANIMAL_CONFIGS,
    MODELS_CONFIGS,
)


@dataclass
class Params:
    video_name: str
    video_path: str
    max_det: int
    enable_flow: bool
    segm_model_id: str
    pose_model_id: str
    flow_model_id: Optional[str]


@dataclass
class TrackTask:
    priority: int
    params: Params
    video_id: int
    user_id: int
    code: Optional[int]
    animal: str


@dataclass
class TrainDetTask:
    priority: int
    config_path: Path
    dataset_id: int
    user_id: int
    val_ratio: float
    animal_name: str
    dataset_root: Path
    model_id: int


@dataclass
class TrainPoseTask:
    priority: int
    config_path: Path
    dataset_id: int
    dataset_root: Path
    user_id: int
    val_ratio: float
    animal_name: str
    skeleton: list[tuple[str, str]]
    keypoints: list[str]
    swap: list[tuple[str, str]]
    model_id: int


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost")
if REDIS_URL.startswith("redis://"):
    REDIS_URL = REDIS_URL[8:]
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
redis = Redis(REDIS_URL, REDIS_PORT)
task_queue = Queue(connection=redis)


resource_path = os.environ.get("RESOURCE_PATH", "static")
model_path = Path("./trained_models/configs")


def build_det_cfg(config_file_path, classes, dataset_root, max_epochs, work_dir: str):
    from mmengine.config import Config

    batch_size = 6
    cfg = Config.fromfile(config_file_path)
    cfg.train_cfg.max_epochs = max_epochs
    cfg.train_cfg.dynamic_intervals = [(max_epochs - 10, 1)]
    if "rcnn" not in config_file_path:
        cfg.custom_hooks[1]["switch_epoch"] = max_epochs - 10

    cfg.work_dir = work_dir
    cfg.train_dataloader.dataset.dataset["metainfo"] = dict()
    cfg.train_dataloader.dataset.dataset["metainfo"]["classes"] = classes
    cfg.val_dataloader.dataset.metainfo["classes"] = classes
    cfg.test_dataloader.dataset.metainfo["classes"] = classes

    cfg.val_dataloader.dataset.data_root = dataset_root
    cfg.test_dataloader.dataset.data_root = dataset_root
    cfg.train_dataloader.dataset.dataset.data_root = dataset_root

    cfg.train_dataloader.dataset.dataset["data_prefix"] = dict(img="")
    cfg.val_dataloader.dataset["data_prefix"] = dict(img="")
    cfg.test_dataloader.dataset["data_prefix"] = dict(img="")

    cfg.train_dataloader.dataset.dataset.ann_file = "train.json"
    cfg.val_dataloader.dataset.ann_file = "val.json"
    cfg.test_dataloader.dataset.ann_file = "val.json"

    cfg.optim_wrapper.type = "AmpOptimWrapper"
    cfg.optim_wrapper.loss_scale = "dynamic"
    cfg.train_dataloader.batch_size = batch_size
    cfg.val_dataloader.batch_size = batch_size
    cfg.test_dataloader.batch_size = batch_size
    dataset_root = Path(dataset_root).resolve()
    cfg.val_evaluator.ann_file = str(dataset_root / "val.json")
    cfg.test_evaluator.ann_file = str(dataset_root / "val.json")
    cfg.auto_scale_lr.base_batch_size = batch_size
    return cfg


class NamedTemporaryFileWithExtension:
    def __init__(self, suffix="", delete=True):
        self.suffix = suffix
        self.delete = delete
        self.file = None
        self.file_path = None

    def __enter__(self):
        # 创建一个临时文件
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.file_path = self.file.name + self.suffix
        self.file.close()
        os.rename(self.file.name, self.file_path)
        # 重新打开文件进行操作
        self.file = open(self.file_path, "w+b")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if self.delete:
            os.remove(self.file_path)


def train_det(task: TrainDetTask):
    from mmengine.runner import Runner

    generate_coco_dataset(
        task.dataset_id, task.dataset_root, task.animal_name, 18, 1 - task.val_ratio
    )
    cfg = build_det_cfg(
        os.path.join(model_path, task.config_path),
        (task.animal_name,),
        str(task.dataset_root),
        100,
        os.path.join(resource_path, f"models/{task.user_id}/{task.model_id}"),
    )

    trainer = Runner.from_cfg(cfg)
    trainer.train()

    from app.api.models.models import Model

    session = sessionmaker(bind=engine)()
    model = session.query(Model).filter(Model.id == task.model_id).first()
    model.trained = True
    session.commit()
    session.close()

    # import subprocess
    # with NamedTemporaryFileWithExtension(suffix=".py", delete=True) as temp_file:
    #     cfg.dump(temp_file.name)
    #     subprocess.run(
    #         f"mim train mmdet {temp_file.name}".split(), capture_output=True, text=True
    #     )


def build_dataset_info(task: TrainPoseTask):
    dataset_info = {
        "dataset_name": f"coco_{task.animal_name}",
        "keypoint_info": {},
        "skeleton_info": {},
        "joint_weights": [1] * len(task.keypoints),
        "sigmas": [0.1] * len(task.keypoints),
    }
    for idx, kpt in enumerate(task.keypoints):
        swap = ""
        for pair in task.swap:
            if kpt in pair:
                swap = set(pair) - set([kpt])
                assert len(swap) == 1, "Invalid swap"
                swap = swap.pop()
                break
        dataset_info["keypoint_info"][idx] = {
            "name": kpt,
            "id": idx,
            "color": [51, 153, 255],
            "type": "",
            "swap": swap,
        }
    for idx, conn in enumerate(task.skeleton):
        dataset_info["skeleton_info"][idx] = {
            "link": conn,
            "id": idx,
            "color": [0, 255, 0],
        }
    return dataset_info


def build_pose_cfg(
    config_file_path, classes, dataset_root, num_kpts, work_dir: str, skeleton_info
):
    from mmengine.config import Config

    batch_size = 6
    cfg = Config.fromfile(config_file_path)
    if "rcnn" in str(config_file_path):
        cfg.model.roi_head.bbox_head.num_classes = len(classes)
        cfg.model.roi_head.mask_head.num_classes = len(classes)
    cfg.model.head.out_channels = num_kpts
    cfg.work_dir = work_dir
    cfg.train_dataloader.dataset.metainfo = skeleton_info
    cfg.val_dataloader.dataset.metainfo = skeleton_info
    cfg.test_dataloader.dataset.metainfo = skeleton_info

    cfg.train_dataloader.dataset.data_root = dataset_root
    cfg.val_dataloader.dataset.data_root = dataset_root
    cfg.test_dataloader.dataset.data_root = dataset_root
    cfg.train_dataloader.dataset.ann_file = "train.json"
    cfg.val_dataloader.dataset.ann_file = "val.json"
    cfg.val_dataloader.dataset.ann_file = "val.json"
    cfg.train_dataloader.dataset["data_prefix"] = dict(img="")
    cfg.val_dataloader.dataset["data_prefix"] = dict(img="")
    cfg.test_dataloader.dataset["data_prefix"] = dict(img="")
    cfg.train_dataloader.batch_size = batch_size * 2
    cfg.auto_scale_lr.base_batch_size = batch_size * 2

    cfg.val_evaluator.ann_file = os.path.join(dataset_root, "val.json")
    cfg.test_evaluator.ann_file = os.path.join(dataset_root, "val.json")
    cfg.model.head.out_channels = num_kpts
    return cfg


def generate_coco_dataset(
    dataset_id, dump_dir: Path, animal_name, num_kpts: int, train_ratio: float = 0.8
):
    from app.api.datasets.models import Image, Annotation

    random.seed(42)

    session = sessionmaker(bind=engine)()
    images = images = (
        session.query(Image)
        .join(Annotation)
        .filter(Image.dataset_id == dataset_id)
        .options(joinedload(Image.annotations))
    )

    image_data = []
    annotation_data = []

    for image in images:
        if image.annotations:  # 确保图像有注释
            image_data.append(
                {
                    "id": image.id,
                    "file_name": image.name,
                    "width": image.width,  # 需要从图像中提取实际宽度
                    "height": image.height,  # 需要从图像中提取实际高度
                }
            )
            for annotation in image.annotations:
                bbox = literal_eval(annotation.bbox) if annotation.bbox else []
                annotation_data.append(
                    {
                        "id": annotation.id,
                        "image_id": annotation.image_id,
                        "category_id": 1,  # 假设只有一个类别，你可以根据需要调整
                        "keypoints": (
                            literal_eval(annotation.keypoints)
                            if annotation.keypoints
                            else []
                        ),
                        "area": annotation.area,
                        "bbox": bbox,
                        "segmentation": (
                            literal_eval(annotation.polygon)
                            if annotation.polygon
                            else []
                        ),
                        "iscrowd": 0,
                        "num_keypoints": num_kpts,
                    }
                )
    session.close()

    random.shuffle(image_data)
    train_split = int(train_ratio * len(image_data))
    train_images = image_data[:train_split]
    val_images = image_data[train_split:]

    train_image_ids = {img["id"] for img in train_images}
    val_image_ids = {img["id"] for img in val_images}

    train_annotations = [
        anno for anno in annotation_data if anno["image_id"] in train_image_ids
    ]
    val_annotations = [
        anno for anno in annotation_data if anno["image_id"] in val_image_ids
    ]

    coco_train = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": [{"id": 1, "name": animal_name, "supercategory": "animal"}],
    }

    coco_val = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": [{"id": 1, "name": animal_name, "supercategory": "animal"}],
    }

    # 将字典转换为JSON并保存到文件
    with open(dump_dir / "train.json", "w") as train_file:
        json.dump(coco_train, train_file)

    with open(dump_dir / "val.json", "w") as val_file:
        json.dump(coco_val, val_file)


def train_pose(task: TrainPoseTask):
    from mmengine.runner import Runner

    generate_coco_dataset(
        task.dataset_id,
        task.dataset_root,
        task.animal_name,
        len(task.keypoints),
        1 - task.val_ratio,
    )
    dataset_info = build_dataset_info(task)
    cfg = build_pose_cfg(
        os.path.join(model_path, task.config_path),
        (task.animal_name,),
        str(task.dataset_root),
        len(task.keypoints),
        os.path.join(resource_path, f"models/{task.user_id}/{task.model_id}"),
        dataset_info,
    )
    runner = Runner.from_cfg(cfg)
    runner.train()

    from app.api.models.models import Model

    session = sessionmaker(bind=engine)()
    model = session.query(Model).filter(Model.id == task.model_id).first()
    model.trained = True
    session.commit()
    session.close()


def generate_json(session, task: TrackTask, video):
    from app.algorithm.models import Datum, TrackletStat

    subquery = (
        session.query(Datum.raw_track_id, Datum.track_id)
        .filter(Datum.video_id == video.id)
        .distinct()
        .subquery()
    )
    tracklets = (
        session.query(TrackletStat, subquery.c.track_id)
        .filter(TrackletStat.video_id == video.id)
        .join(subquery, TrackletStat.track_id == subquery.c.raw_track_id)
    )

    data = session.query(Datum).filter(
        Datum.video_id == task.video_id,
        Datum.track_id <= task.params.max_det,
        Datum.track_id > 0,
    )

    try:
        from app.api.models.models import Model

        model_id = int(task.params.pose_model_id)
        model = session.query(Model).filter(Model.id == model_id).first()
        assert model is not None, "Model not found"
        params = literal_eval(model.params)
        kpts = params["keypoints"]
        nkpts = len(kpts)
        connections = []
        for link in params["link"]:
            connections.append([kpts.index(link[0]), kpts.index(link[1])])
    except ValueError:
        nkpts = ANIMAL_CONFIGS[task.animal]["nkpts"]
        connections = ANIMAL_CONFIGS[task.animal]["connections"]

    keypoints = np.zeros((task.params.max_det, video.frame_cnt, nkpts, 2))
    for d in data:
        keypoints[d.track_id - 1, d.frame] = literal_eval(d.keypoints)
    json_data = {
        "headers": {
            "connections": connections,
            "interval": [0, video.frame_cnt - 1],
            "tracklets": [
                {
                    "intervals": literal_eval(t.intervals),
                    "rawTrackID": t.track_id,
                    "trackID": track_id,
                }
                for t, track_id in tracklets
            ],
        },
    }
    json_data["data"] = keypoints.tolist()
    token, ext = os.path.splitext(video.name)
    json_path = Path(f"{resource_path}/json/{task.user_id}/{token}.json")
    if not json_path.parent.exists():
        json_path.parent.mkdir(parents=True)
    with open(json_path, "w") as f:
        json.dump(json_data, f)


def get_config_checkpoint(model_root: Path):
    config, checkpoint = None, None
    for model_file in model_root.glob("*.pth"):
        if "best" in model_file.stem:
            checkpoint = model_file
            break
    for mconfig_file in model_root.glob("*.py"):
        config = mconfig_file
        break
    return str(config), str(checkpoint)


def track(task: TrackTask):
    from app.api.videos.models import Video
    from app.api.models.models import Model
    from app.algorithm.stage1 import online_track
    from app.algorithm.stage2 import merge

    session = sessionmaker(bind=engine)()
    video = session.query(Video).filter(Video.id == task.video_id).first()
    video.analyzed = False
    session.commit()
    try:
        segm_id = int(task.params.segm_model_id)
        det_model = session.query(Model).filter(Model.id == segm_id).first()
        assert det_model is not None, "Model not found"
        assert det_model.category == "segm", "Model is not a segmentation model"
        model_root = (
            Path(resource_path) / "models" / str(task.user_id) / str(det_model.id)
        )
        det_config_file, det_checkpoint_file = get_config_checkpoint(model_root)
        assert det_config_file is not None, "Config file not found"
        assert det_checkpoint_file is not None, "Checkpoint file not found"
    except ValueError:
        det_config_file = MODELS_CONFIGS[task.params.segm_model_id]["config"]
        det_checkpoint_file = MODELS_CONFIGS[task.params.segm_model_id]["checkpoint"]

    try:
        pose_id = int(task.params.pose_model_id)
        pose_model = session.query(Model).filter(Model.id == pose_id).first()
        assert pose_model is not None, "Model not found"
        assert pose_model.category == "pose", "Model is not a pose model"
        model_root = (
            Path(resource_path) / "models" / str(task.user_id) / str(pose_model.id)
        )
        pose_config_file, pose_checkpoint_file = get_config_checkpoint(model_root)
        assert pose_config_file is not None, "Config file not found"
        assert pose_checkpoint_file is not None, "Checkpoint file not found"
    except ValueError:
        pose_config_file = MODELS_CONFIGS[task.params.pose_model_id]["config"]
        pose_checkpoint_file = MODELS_CONFIGS[task.params.pose_model_id]["checkpoint"]
    if task.params.enable_flow:
        flow_config_file = MODELS_CONFIGS[task.params.flow_model_id]["config"]
        flow_checkpoint_file = MODELS_CONFIGS[task.params.flow_model_id]["checkpoint"]
    else:
        flow_config_file = ""
        flow_checkpoint_file = ""

    online_track(
        resource_path,
        task.video_id,
        MAX_WIDTH_HEIGHT,
        task.params.max_det,
        DET_BATCH_SIZE,
        task.params.enable_flow,
        CONF_THRESH,
        NMS_THRESH,
        IOU_THRESHOLD,
        MAX_AGE,
        det_config_file,
        det_checkpoint_file,
        pose_config_file,
        pose_checkpoint_file,
        flow_config_file,
        flow_checkpoint_file,
    )
    merge(
        resource_path,
        task.video_id,
        task.params.max_det,
        CLS_CONFIG_FILE,
        CLS_CHECKPOINT_FILE,
        BATCH_SIZE,
        256,
        0.9,
        0.6,
        SOFT_BORDER,
    )
    # cmd = [
    #     f"{sys.executable}",
    #     "track_stage1.py",
    #     task.params.video_name,
    #     task.params.video_path,
    #     str(task.params.max_det),
    # ]
    # if task.params.enable_flow:
    #     cmd.append("--enable_flow")
    # return_code = subprocess.call(cmd)
    # if return_code != 0:
    #     raise Exception("Error occurred while performing tracking stage 1")
    # return_code = 0

    # cmd = [
    #     f"{sys.executable}",
    #     "track_stage2.py",
    #     task.params.video_name,
    #     task.params.video_path,
    #     str(task.params.max_det),
    # ]
    # # cmd.extend(["--resume", "28"])
    # return_code = subprocess.call(cmd)
    # if return_code != 0:
    #     raise Exception("Error occurred while performing tracking stage 2")
    video.analyzed = True
    session.commit()
    generate_json(session, task, video)
    session.close()
    return 0


import subprocess
import tempfile
import os


def run_ffmpeg_with_progress(input_file, output_file):
    # Create a temporary file to store progress info
    import time

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    # Initialize FFmpeg subprocess with -progress
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-preset",
        "ultrafast",
        "-progress",
        temp_file_path,
        output_file,
    ]
    process = subprocess.Popen(
        cmd, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True
    )

    while True:
        # Read the temporary file to get progress
        time.sleep(0.5)
        with open(temp_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "out_time_ms" in line:
                    out_time_ms = int(line.split("=")[1].strip())
                    print(f"Progress: {out_time_ms / 1000000.0} seconds")

        if process.poll() is not None:
            break

    # Remove the temporary file
    os.remove(temp_file_path)

    process.wait()


if __name__ == "__main__":
    input_file = "/mnt/e/projects/videos/6xWT.MOV"
    output_file = "/mnt/e/projects/videos/testvideos/4mice1_out.mp4"
    run_ffmpeg_with_progress(input_file, output_file)
