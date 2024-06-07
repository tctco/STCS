from pathlib import Path
import shutil
import cv2
import mmcv
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from app.algorithm.det import MMDetImageDetector, YOLODetector
from app.algorithm.pose import YOLOPoseBottomUpEstimator, MMPoseTopDownEstimator
from app.algorithm.tracker import (
    FlowTracker,
    MaskTracker,
    MMFlowEstimator,
    OpenCVFlowEstimator,
)
from app.common.database import engine
from app.algorithm.models import Datum, TrackletStat
from app.api.videos.models import Video
from app.algorithm.common import create_logger, set_job_meta


def batch_frame_generator(video: mmcv.VideoReader, batch_size=1, max_frame=-1):
    batch = []
    frame_indexes = []
    for i, frame in enumerate(tqdm(video)):
        max_frame = max_frame if max_frame > 0 else len(video)
        if i >= max_frame:
            yield batch, frame_indexes
            batch, frame_indexes = [], []
            break
        batch.append(frame)
        frame_indexes.append(i)
        if i % batch_size == batch_size - 1:
            set_job_meta("progress", i / max_frame)
            yield batch, frame_indexes
            batch, frame_indexes = [], []
    if len(batch) > 0:
        set_job_meta("progress", 1)
        yield batch, frame_indexes


def log_configs(logger, **kwargs):
    logger.info("Config:")
    for k, v in kwargs.items():
        logger.info(f"{k}: {v}")


def online_track(
    resource_path: str,
    video_id: int,
    max_width_height: int,
    max_det: int,
    batch_size,
    enable_flow: bool,
    conf_threshold: float,
    nms_threshold: float,
    iou_threshold: float,
    max_age: int,
    det_config_file: str,
    det_checkpoint_file: str,
    pose_config_file: str,
    pose_checkpoint_file: str,
    flow_config_file: str,
    flow_checkpoint_file: str,
    max_frame=-1,
):
    session = sessionmaker(bind=engine)()
    video = session.query(Video).filter(Video.id == video_id).first()
    assert video is not None, f"Video with id {video_id} not found"
    session.query(Datum).filter(Datum.video_id == video_id).delete()
    session.query(TrackletStat).filter(TrackletStat.video_id == video_id).delete()

    file_save_path = Path(
        f"{resource_path}/exp/{video.user_id}/{video.name.split('.')[0]}_{video.original_name.split('.')[0]}"
    )
    if file_save_path.exists():
        shutil.rmtree(file_save_path)
    file_save_path.mkdir(parents=True, exist_ok=True)
    logger = create_logger("stage1", str(file_save_path / "stage1.log"))
    set_job_meta("stage", 1)
    log_configs(
        logger,
        video_id=video_id,
        video_name=video.original_name,
        max_width_height=max_width_height,
        max_det=max_det,
        batch_size=batch_size,
        enable_flow=enable_flow,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        iou_threshold=iou_threshold,
        max_age=max_age,
        det_config_file=det_config_file,
        det_checkpoint_file=det_checkpoint_file,
        pose_config_file=pose_config_file,
        pose_checkpoint_file=pose_checkpoint_file,
        flow_config_file=flow_config_file,
        flow_checkpoint_file=flow_checkpoint_file,
        max_frame=max_frame,
    )

    vpath = Path(resource_path) / f"videos/{video.user_id}/{video.name}"
    cap = mmcv.VideoReader(str(vpath))
    if max_width_height > 0 and max_width_height < min(cap.width, cap.height):
        if cap.width > cap.height:
            width = max_width_height
            scale = width / cap.width
            height = int(cap.height * scale)
        else:
            height = max_width_height
            scale = height / cap.height
            width = int(cap.width * scale)
        logger.info(f"Resize video to {width}x{height}")
    else:
        width, height = cap.width, cap.height
        scale = 1

    batch_video_reader = batch_frame_generator(
        cap, batch_size=batch_size, max_frame=max_frame
    )

    if det_config_file == "YOLO":
        detector = YOLODetector(
            max_det=max_det,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            checkpoint_file=det_checkpoint_file,
        )
    else:
        detector = MMDetImageDetector(
            max_det=max_det,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            config_file=det_config_file,
            checkpoint_file=det_checkpoint_file,
        )
    if pose_config_file == "YOLO":
        pose_estimator = YOLOPoseBottomUpEstimator(
            checkpoint=pose_checkpoint_file,
        )
    else:
        pose_estimator = MMPoseTopDownEstimator(
            config=pose_config_file, checkpoint=pose_checkpoint_file
        )

    # TODO: this is ugly, needs to be refactored
    if enable_flow:
        if flow_config_file == "opencv":
            flow_estimator = OpenCVFlowEstimator()
        else:
            flow_estimator = MMFlowEstimator(
                config=flow_config_file,
                checkpoint=flow_checkpoint_file,
            )
        tracker = FlowTracker(
            max_age=max_age, iou_threshold=iou_threshold, w=cap.width, h=cap.height
        )
    else:
        tracker = MaskTracker(
            max_age=max_age, iou_threshold=iou_threshold, w=cap.width, h=cap.height
        )

    for batch, frame_indexes in batch_video_reader:
        if len(batch) == 0:
            break
        batch = [cv2.resize(x, (width, height)) for x in batch]
        detections = detector.predict(batch)
        poses = pose_estimator.predict(batch, detections)

        if enable_flow:
            flows = flow_estimator.predict(batch)
        else:
            flows = [None] * len(batch)

        for frame, flow, pose, frame_idx in zip(batch, flows, poses, frame_indexes):
            if max_det > 1:
                if enable_flow:
                    ids, dead_tracklets = tracker.update(flow, pose, frame_idx)
                    # mmcv.imwrite(
                    #     (mmcv.flow2rgb(flow) * 255).astype(np.uint8),
                    #     str(file_save_path / f"flow/{frame_idx}.jpg"),
                    # )
                else:
                    ids, dead_tracklets = tracker.update(pose, frame_idx)

                for x in dead_tracklets:
                    session.add(x.get_stats(video_id))

                pose.save_images(frame, ids, frame_idx, str(file_save_path))
                data = pose.get_datum(video.id, frame_idx, ids, scale=scale)
            else:
                data = pose.get_datum(video.id, frame_idx, [0], track_id=0, scale=scale)
            for x in data:
                session.add(x)
            session.commit()
    for t in tracker.cached_tracklets:
        session.add(t.get_stats(video_id))
    session.commit()
    logger.info(f"time consumed:")
    logger.info(f"detection {detector.timer.t}")
    logger.info(f"pose {pose_estimator.timer.t}")
    if enable_flow:
        logger.info(f"flow {flow_estimator.timer.t}")
    else:
        logger.info(f"flow disabled")
    logger.info(f"tracking {tracker.timer.t}")
    logger.info("Stage1 finished")
