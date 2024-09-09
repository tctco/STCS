import ast
from pathlib import Path
import os
from typing import Optional
import cv2
import numpy as np
from app.algorithm.merger import Merger, LibraryMerger, Tracklet, Interval
from app.algorithm.classifier import MMPretrainClassifier
from app.algorithm.models import TrackletStat, Datum
from app.algorithm.common import create_logger
from app.api.videos.models import Video
from app.common.database import engine
from sqlalchemy.orm import sessionmaker


def calc_img_shape(img_root, max_crop_size):
    img_shapes = []
    for id in os.listdir(img_root):
        if not os.path.isdir(img_root / id):
            continue
        for file in os.listdir(img_root / id):
            img = cv2.imread(str(img_root / id / file))
            img_shapes.append(img.shape[:2])
    img_shapes = np.array(img_shapes)
    img_shape = np.percentile(img_shapes, 90, axis=0) // 32 * 32 + 32
    if max(img_shape) > max_crop_size:
        scale_factor = max_crop_size / max(img_shape)
        img_shape = int(img_shape[0] * scale_factor), int(img_shape[1] * scale_factor)
    else:
        scale_factor = 1
        img_shape = int(img_shape[0]), int(img_shape[1])
    return img_shape, scale_factor


def convert_intervals(index, intervals):
    assert (
        len(intervals) % 2 == 0
    ), f"intervals should be even, but got {len(intervals)} ({intervals})"
    result = []
    for i in range(0, len(intervals), 2):
        result.append(Interval(index, intervals[i], intervals[i + 1]))
    return result


def merge(
    resource_path,
    video_id,
    max_det,
    cls_config,
    cls_checkpoint,
    batch_size,
    max_crop_size,
    confidence_threshold,
    min_confidence,
    soft_border,
    train_ratio=0.9,
    max_frames_in_model_building: Optional[int] = None,
):
    resource_path = Path(resource_path)
    session = sessionmaker(bind=engine)()
    video = session.query(Video).filter(Video.id == video_id).first()
    exp_root = (
        resource_path
        / f"exp/{video.user_id}/{video.name.split('.')[0]}_{video.original_name.split('.')[0]}"
    )
    logger = create_logger("stage2", str(exp_root / "stage2.log"))
    tracklets = (
        session.query(TrackletStat).filter(TrackletStat.video_id == video_id).all()
    )
    tracklets = [
        Tracklet(convert_intervals(x.track_id, ast.literal_eval(x.intervals)))
        for x in tracklets
    ]
    cls_model_save_path = exp_root / "cls"
    img_root = exp_root / "cropped"
    video_path = resource_path / f"{video.user_id}/{video.name}"
    img_shape, scale = calc_img_shape(img_root, max_crop_size)
    classifier = MMPretrainClassifier(
        cls_config,
        cls_checkpoint,
        str(cls_model_save_path),
        str(img_root),
        batch_size,
        max_det,
        img_shape,
        logger,
        scale,
    )
    if max_frames_in_model_building is None:
        merger = Merger(
            tracklets,
            max_det,
            classifier,
            img_root,
            exp_root,
            video_path,
            logger,
            session,
            video_id,
            confidence_threshold,
            soft_border,
            train_ratio,
            max_frames_per_class=6000,
            batch_size=batch_size,
            min_merge_frames=150,
            apperance_threshold=0.05,
            min_confidence=min_confidence,
        )
    else:
        merger = LibraryMerger(
            max_frames_in_model_building,
            tracklets,
            max_det,
            classifier,
            img_root,
            exp_root,
            video_path,
            logger,
            session,
            video_id,
            confidence_threshold,
            soft_border,
            train_ratio,
            max_frames_per_class=6000,
            batch_size=batch_size,
            min_merge_frames=150,
            apperance_threshold=0.05,
            min_confidence=min_confidence,
        )

    merger.merge()
    tracks = merger.assigned_tracklets
    for i, t in enumerate(tracks):
        ids = t.get_ids()
        # tracklets = (
        #     session.query(TrackletStat).filter(TrackletStat.track_id.in_(ids)).all()
        # )
        # for tracklet in tracklets:
        #     tracklet.track_id = i
        data = session.query(Datum).filter(Datum.raw_track_id.in_(ids)).all()
        for d in data:
            d.track_id = i + 1
    session.commit()
