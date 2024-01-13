from app.algorithm.merger import Merger, Tracklet
from app.algorithm.classifier import MMPretrainClassifier
from app.algorithm.models import TrackletStat
from app.algorithm.common import create_logger
from app.api.videos.models import Video
from sqlalchemy.orm import sessionmaker
from app.common.database import engine
import ast
from pathlib import Path
import os
import cv2
import numpy as np


def calc_img_shape(img_root, max_crop_size):
    img_shapes = []
    for id in os.listdir(img_root):
        if not os.path.isdir(img_root / id):
            continue
        for file in os.listdir(img_root / id):
            img = cv2.imread(img_root / id / file)
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


def merge(
    resource_path,
    video_id,
    max_det,
    cls_config,
    cls_checkpoint,
    batch_size,
    max_crop_size,
    confidence_threshold,
    soft_border,
    train_ratio=0.9,
):
    resource_path = Path(resource_path)
    session = sessionmaker(bind=engine)()
    video = session.query(Video).filter(Video.id == video_id).first()
    tracklets = (
        session.query(TrackletStat).filter(TrackletStat.video_id == video_id).all()
    )
    tracklets = [Tracklet(ast.literal_eval(x.intervals)) for x in tracklets]
    logger = create_logger("stage2")
    exp_root = (
        resource_path
        / f"exp/{video.user_id}/{video.name.split('.')[0]}_{video.original_name.split('.')[0]}"
    )
    cls_model_save_path = exp_root / "cls"
    img_root = exp_root / "cropped"
    video_path = resource_path / f"{video.user_id}/{video.name}"
    img_shape, scale = calc_img_shape(img_root, max_crop_size)
    classifier = MMPretrainClassifier(
        cls_config,
        cls_checkpoint,
        str(cls_model_save_path),
        batch_size,
        max_det,
        img_shape,
        logger,
        scale,
    )
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
        min_confidence=0.6,
    )

    merger.merge()
    tracks = merger.assigned_tracklets
    for i, t in enumerate(tracks):
        ids = set([x.id for x in t.intervals])
        tracklets = session.query(TrackletStat).filter(TrackletStat.id.in_(ids)).all()
        for tracklet in tracklets:
            tracklet.track_id = i
