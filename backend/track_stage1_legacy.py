from mmdet.apis import inference_detector, init_detector
import mmcv
from mmcv.ops import bbox_overlaps
from mmengine.utils import track_iter_progress
import numpy as np
import numba
from ultralytics import YOLO
import pycocotools.mask as maskUtils

# from mmpose.evaluation.functional import nms
from mmcv.ops import soft_nms
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
from mmengine.registry import init_default_scope
import cv2
from typing import List, Tuple
import os
from pathlib import Path
import logging
import datetime
import shutil
import sys
from mmengine.dataset import Compose, pseudo_collate
import torch
from mmdet.utils import get_test_pipeline_cfg
import gc
from mmflow.apis import (
    inference_model as flow_inference,
    init_model as init_flow_estimator,
)
from sqlalchemy.orm import sessionmaker
from argparse import ArgumentParser

from constants import *
from app.algorithm.models import Datum, TrackletStat
from app.algorithm.timer import Profile
from app.common.database import engine
from app.api.videos.models import Video

parser = ArgumentParser()
parser.add_argument("video_name", help="video name without extension", type=str)
parser.add_argument("video_path", help="video root path", type=str)
parser.add_argument("max_det", help="maximum animals in the video", type=int)
parser.add_argument("--first", help="first N frames", type=int)
parser.add_argument("--enable_flow", help="enable flow", action="store_true")
parser.add_argument("--job-id", help="rq worker job id", type=str)
args = parser.parse_args()
if args.video_name is not None:
    VIDEO_NAME = args.video_name
if args.video_path is not None:
    VIDEO_PATH = f"{args.video_path}/{VIDEO_NAME}.mp4"
if args.max_det is not None:
    MAX_DET = args.max_det
if args.job_id is not None:
    from redis import Redis
    from rq.job import Job

    current_job_id = args.job_id
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost")
    if REDIS_URL.startswith("redis://"):
        REDIS_URL = REDIS_URL[8:]
    REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
    redis = Redis(REDIS_URL, REDIS_PORT)
    job = Job.fetch(current_job_id, connection=redis)
else:
    job = None
ENABLE_FLOW = args.enable_flow
if not ENABLE_FLOW:
    DET_BATCH_SIZE += 4

log_dir = Path("./exp/logs")
if not log_dir.exists():
    log_dir.mkdir(parents=True)

str_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
logging.basicConfig(
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"./exp/logs/{VIDEO_NAME}_test_{str_time}.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

logging.info(f"current job id: {job.id if job is not None else None}")
logging.info(
    f"flow enabled: {ENABLE_FLOW}, max age: {MAX_AGE}, max det: {MAX_DET}, det batch size {DET_BATCH_SIZE}, conf threshold: {CONF_THRESH}, IOU threshold: {IOU_THRESHOLD}, upper kpt: {UPPER_KEYPOINT}, lower kpt: {LOWER_KEYPOINT}"
)
logging.info(f"det config file: {DET_CONFIG_FILE}, det ckpt: {DET_CHECKPOINT_FILE}")
logging.info(f"pose config file: {POSE_CONFIG_FILE}, pose ckpt: {POSE_CHECKPOINT_FILE}")


exp_path = Path(f"exp/files/{VIDEO_NAME}")
exp_path.mkdir(parents=True, exist_ok=True)

pose_timer = Profile()
det_timer = Profile()
viz_timer = Profile()
crop_timer = Profile()
flow_timer = Profile()
noneed_timer = Profile()

gc.collect()
torch.cuda.empty_cache()


# root_path = Path(f"./vipseg")

# deva_img_path = root_path / "images" / VIDEO_NAME
# if deva_img_path.exists():
#     shutil.rmtree(deva_img_path)
# deva_img_path.mkdir(parents=True)
# deva_mask_path = root_path / "source" / VIDEO_NAME
# if deva_mask_path.exists():
#     shutil.rmtree(deva_mask_path)
# deva_mask_path.mkdir(parents=True)


# def save_masks_json(frame, img, masks, scores):
#     from panopticapi.utils import IdGenerator

#     vipseg_idgen = IdGenerator({1: {"isthing": 1, "color": [50, 0, 0]}})
#     json_data = []
#     cv2.imwrite(str(deva_img_path / f"{frame:0>8}.jpg"), img)
#     empty = np.zeros(img.shape, dtype=np.uint8)
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         segid, color = vipseg_idgen.get_id_and_color(1)
#         empty[mask == 1, :] = color
#         json_data.append(
#             {
#                 "id": segid,
#                 "isthing": True,
#                 "category_id": 1,
#                 "score": round(score.item(), 2),
#             }
#         )
#     cv2.imwrite(str(deva_mask_path / f"{frame:0>8}.png"), mmcv.rgb2bgr(empty))
#     with open(str(deva_mask_path / f"{frame:0>8}.json"), "w") as f:
#         json.dump(json_data, f)


def random_color(seed):
    """Random a color according to the input seed."""
    np.random.seed(seed)
    color = (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
    )
    return color


def masks_iou(masks1: np.ndarray, masks2: np.ndarray) -> float:
    return (masks1 & masks2).sum() / (masks1 | masks2).sum()


@numba.jit(nopython=True)
def iom(mask1, mask2):
    return (mask1 & mask2).sum() / max(mask1.sum(), mask2.sum())


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def match(cost_matrix: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    matches = linear_assignment(cost_matrix)
    q, o = cost_matrix.shape
    unmatched_q = list(set(range(q)) - set([i for i, j in matches]))
    unmatched_o = list(set(range(o)) - set([j for i, j in matches]))
    return matches, unmatched_q, unmatched_o


class Tracklet:
    count: int = 0

    def __init__(
        self,
        frame_idx: int,
        mask: np.ndarray,
        centroid: Tuple[float, float],
        bbox: np.ndarray,
        conf: float,
    ) -> None:
        self.id: int = Tracklet.count
        Tracklet.count += 1
        self.last_mask: np.ndarray = mask
        self.last_centroid: Tuple[float, float] = centroid
        self.last_bbox: np.ndarray = bbox
        self.start_frame: int = frame_idx
        self.last_frame: int = frame_idx
        self.max_age: int = MAX_AGE
        self.intervals = []
        self.conf = [conf]
        x1, y1, x2, y2 = parse_bbox(bbox)
        self.mask_area = [mask[y1:y2, x1:x2].sum()]
        self.distance = 0

    def update(
        self,
        mask: np.ndarray,
        centroid: Tuple[float, float],
        bbox: np.ndarray,
        frame_idx: int,
        conf: float,
    ):
        if frame_idx > self.last_frame + 1:
            self.intervals += [self.last_frame, frame_idx]
        self.last_mask = mask
        self.last_frame = frame_idx
        self.distance += np.linalg.norm(
            np.array(centroid) - np.array(self.last_centroid)
        )
        self.last_centroid = centroid
        self.last_bbox = bbox
        self.conf.append(conf)
        x1, y1, x2, y2 = parse_bbox(bbox)
        self.mask_area.append(mask[y1:y2, x1:x2].sum())

    def get_stats(self):
        return TrackletStat(
            video_id=VIDEO_ID,
            track_id=self.id,
            start_frame=self.start_frame,
            end_frame=self.last_frame,
            lifespan=self.last_frame - self.start_frame + 1,
            intervals=str([self.start_frame, *self.intervals, self.last_frame]),
            conf=np.mean(self.conf),
            mask_area=np.mean(self.mask_area),
            distance=self.distance,
        )


def check_valid_tracks(
    tracks: List[Tracklet], frame_idx: int
) -> Tuple[List[Tracklet], List[Tracklet]]:
    valid_tracks = []
    stale_tracks = []
    for track in tracks:
        if frame_idx - track.max_age > track.last_frame:
            stale_tracks.append(track)
        else:
            valid_tracks.append(track)
    return valid_tracks, stale_tracks


# @numba.jit(nopython=True)
def create_rotation_mat(theta: float):
    """Create a rotation matrix for a given angle in degrees."""
    theta_r = -np.radians(theta)
    R = np.array(
        ((np.cos(theta_r), -np.sin(theta_r)), (np.sin(theta_r), np.cos(theta_r)))
    )
    return R


# @numba.jit()
def rotate_image(img, M, pivot):
    # https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
    expanded_img_shape = list(img.shape[:2])
    expanded_img_shape[0] += pivot[1]
    expanded_img_shape[1] += pivot[0]
    expanded_img_shape = tuple(expanded_img_shape)
    dst = cv2.warpAffine(
        img,
        M,
        expanded_img_shape,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return dst


def get_shift(h, w, center, angle) -> Tuple[np.ndarray, np.ndarray]:
    vec = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    R = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = R @ vec
    shift = abs(rotated.min(axis=1)).round().astype(int)
    rotated += shift[:, None]
    newsize = rotated.max(axis=1).round().astype(int)
    return shift, newsize


def add_homo_coord_kpts(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((points.shape[0], 1))])


def crop_minAreaRect(img, rect, pose, ismask: bool):
    """Crop a rotated rectangle from an image with pose info."""
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int
    center, size = list(map(int, center)), list(map(int, size))
    shift, newsize = get_shift(img.shape[0], img.shape[1], center, theta)
    padlist = [(shift[1], 0), (shift[0], 0), (0, 0)]
    pad_val = 255
    if ismask:
        padlist = padlist[:2]
        pad_val = 0
    padded_img = np.pad(img, padlist, mode="constant", constant_values=pad_val)
    center[0] += shift[0].item()
    center[1] += shift[1].item()
    R = cv2.getRotationMatrix2D(center, theta, 1)
    shifted_pose = pose + shift.reshape((1, 2))
    homogeneous_pose = add_homo_coord_kpts(shifted_pose)
    rotated_pose = (R @ homogeneous_pose.T).T
    rotated_pose[pose < 0] = 0
    rotated_img = cv2.warpAffine(
        padded_img,
        R,
        newsize,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=pad_val,
    )
    left_upper_corner = np.array(center) - np.array(size) // 2
    rotated_pose -= left_upper_corner
    out = cv2.getRectSubPix(rotated_img, size, center)
    center = np.array([out.shape[1], out.shape[0]]) / 2
    if size[0] > size[1]:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        rotated_pose = rotated_pose[:, ::-1]
        rotated_pose[:, 0] = out.shape[1] - rotated_pose[:, 0]
    if rotated_pose[0, 1] >= 0 and rotated_pose[0, 1] > rotated_pose[5, 1]:
        out = cv2.rotate(out, cv2.ROTATE_180)
        R = cv2.getRotationMatrix2D(center, 180, 1)
        rotated_pose = (R @ (add_homo_coord_kpts(rotated_pose).T)).T
    return out, rotated_pose


def convert_datasamples_to_dict(data_sample, cropped_images):
    pred_instances = data_sample.pred_instances
    res = {"pred_instances": {}, "metainfo": data_sample.metainfo}
    for k, v in pred_instances.items():
        if k == "cropped_images":
            res["pred_instances"][k] = cropped_images
        elif k == "masks":
            continue
        else:
            res["pred_instances"][k] = v
    return res


def yolo_predict(frame, det_model):
    with det_timer:
        dets = det_model(frame, max_det=MAX_DET, conf=CONF_THRESH)[0]
    bboxes = dets.boxes.boxes.cpu().numpy()
    scores = bboxes[:, 4]
    bboxes = bboxes[:, :4]
    _masks = dets.masks.masks.cpu().numpy().astype(np.uint8)
    dsize = frame.shape[:2][::-1]
    masks = [
        cv2.resize(m, dsize=dsize, interpolation=cv2.INTER_NEAREST) for m in _masks
    ]
    masks = np.array(masks)
    return bboxes, scores, masks


def process_det_pred_instances(pred_instances):
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy().astype(np.uint8)
    # bboxes = np.concatenate(
    # (bboxes, scores[:, None]), axis=1)

    _, nms_result = soft_nms(bboxes, scores, NMS_THRESH)
    # _, nms_result = batched_nms(bboxes, scores, {'iou_threshold':NMS_THRESH})
    bboxes = bboxes[nms_result]
    masks = masks[nms_result]
    scores = scores[nms_result]

    scores_result = scores > CONF_THRESH
    bboxes = bboxes[scores_result]
    masks = masks[scores_result]
    scores = scores[scores_result]
    if len(bboxes) > MAX_DET:
        bboxes = bboxes[:MAX_DET]
        masks = masks[:MAX_DET]
        scores = scores[:MAX_DET]
    # assert len(bboxes) == len(masks) and len(masks) == len(scores)
    return bboxes, scores, masks


def process_det_result(det):
    pred_instances = det.get("pred_instances", None)
    if pred_instances is None:
        bboxes, scores, masks = np.array([]), np.array([]), np.array([])
    else:
        bboxes, scores, masks = process_det_pred_instances(pred_instances)
    return bboxes, scores, masks


def rtmdet_predict(frame, det_model):
    if isinstance(frame, (List, Tuple)):
        is_batch = True
    else:
        is_batch = False
    init_default_scope(det_model.cfg.get("default_scope", "mmdet"))
    bboxes_list, scores_list, masks_list = [], [], []
    if is_batch:
        cfg = det_model.cfg
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        test_pipeline[0].type = "mmdet.LoadImageFromNDArray"
        test_pipeline = Compose(test_pipeline)
        data = [{"img": x, "img_id": i} for i, x in enumerate(frame)]
        data = [test_pipeline(x) for x in data]
        data[0]["inputs"] = [x["inputs"] for x in data]
        data[0]["data_samples"] = [x["data_samples"] for x in data]
        with det_timer:
            with torch.no_grad():
                det_result = det_model.test_step(data[0])
        for det in det_result:
            bboxes, scores, masks = process_det_result(det)
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            masks_list.append(masks)
        return bboxes_list, scores_list, masks_list
    else:
        with det_timer:
            det_result = inference_detector(det_model, frame)
        return process_det_result(det_result)


def enlarge_bboxes(bboxes, maxh, maxw, ratio=0.1):
    larger_bboxes = np.array(bboxes).reshape((-1, 4))
    x1, y1, x2, y2 = (
        larger_bboxes[:, 0],
        larger_bboxes[:, 1],
        larger_bboxes[:, 2],
        larger_bboxes[:, 3],
    )
    w, h = x2 - x1, y2 - y1
    x1 -= w * ratio
    y1 -= h * ratio
    x2 += w * ratio
    y2 += h * ratio
    larger_bboxes = np.stack((x1, y1, x2, y2), axis=1)
    larger_bboxes[larger_bboxes < 0] = 0
    larger_bboxes[:, 2] = np.clip(larger_bboxes[:, 2], 0, maxw)
    larger_bboxes[:, 3] = np.clip(larger_bboxes[:, 3], 0, maxh)
    return larger_bboxes


def parse_bbox(bbox) -> list[int]:
    bbox = np.array(bbox)
    larger_bbox = enlarge_bboxes(bbox, video.height, video.width)[0]
    return list(map(round, larger_bbox))


def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, 1, 1)
    c = max(contours, key=cv2.contourArea)
    # contours = np.concatenate(contours)
    # c = cv2.convexHull(contours)
    return c


# def compute_centroid(contour):
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         centroid_x = int(M["m10"] / M["m00"])
#         centroid_y = int(M["m01"] / M["m00"])
#     else:
#         centroid_x, centroid_y = 0, 0
#     return centroid_x, centroid_y


def compute_centroid(mask):
    ys, xs = np.where(mask == 1)
    return np.mean(xs), np.mean(ys)


def batch_topdown(model, imgs_list, bboxes_list, scores_list, masks_list):
    assert len(imgs_list) == len(bboxes_list) and len(imgs_list) == len(
        masks_list
    ), "imgs and bboxes should have the same length"

    if isinstance(pose_model, YOLO):
        from mmpose.structures import PoseDataSample
        from mmengine.structures import InstanceData

        img_list = [mmcv.bgr2rgb(img) for img in imgs_list]
        results = pose_model(img_list, verbose=False)
        preds = []
        for poses, bboxes, scores, masks in zip(
            results, bboxes_list, scores_list, masks_list
        ):
            pred_bboxes = poses.boxes.data.cpu().numpy()[:, :4]
            iou_mat = bbox_overlaps(
                torch.tensor(bboxes), torch.tensor(pred_bboxes)
            ).numpy()
            if len(bboxes) == 0 or len(pred_bboxes) == 0:
                preds.append([])
                continue
            matches = linear_assignment(-iou_mat)
            keypoints = poses.keypoints.data.cpu().numpy()
            tmp = []
            for i, j in matches:
                inst = InstanceData()
                inst.bboxes = bboxes[i][None]
                inst.masks = masks[i][None]
                inst.bbox_scores = np.ones(1) * scores[i]
                inst.keypoints = keypoints[j, :, :2][None]
                inst.keypoint_scores = keypoints[j, :, 2][None]
                inst.keypoints_visible = keypoints[j, :, 2][None]
                inst = PoseDataSample(pred_instances=inst)
                tmp.append(inst)
            preds.append(tmp)
        return preds
    else:
        init_default_scope(model.cfg.get("default_scope", "mmpose"))
        pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
        data_list = []
        for img, bboxes, scores, masks in zip(
            imgs_list, bboxes_list, scores_list, masks_list
        ):
            # bboxes = enlarge_bboxes(bboxes, img.shape[0], img.shape[1])
            for box, score in zip(bboxes, scores):
                data_info = dict(
                    img=img,
                    bbox=box[None],
                    bbox_score=np.ones(1) * score,
                )
                data_info.update(model.dataset_meta)
                data_list.append(pipeline(data_info))
        if data_list:
            batch = pseudo_collate(data_list)
            with torch.no_grad():
                preds = model.test_step(batch)
            lengths = [len(l) for l in bboxes_list]
            results = []
            for length, masks in zip(lengths, masks_list):
                results.append(preds[:length])
                for pred, mask in zip(results[-1], masks):
                    pred.pred_instances.masks = mask[None]
                preds = preds[length:]
            return results
        else:
            return []


def process_frame(frame, frame_idx_list, det_model, pose_model):
    """Process a single frame.

    Args:
      frame: The frame to process in BGR.
    """
    if DET_MODEL == "YOLO":
        bboxes, scores, masks = yolo_predict(frame, det_model)
    else:
        if isinstance(frame, (List, Tuple)):
            is_batch = True
        else:
            is_batch = False
        if is_batch:
            bboxes_list, scores_list, masks_list = rtmdet_predict(frame, det_model)
            frame_list = frame
            with pose_timer:
                poses_list = batch_topdown(
                    pose_model, frame_list, bboxes_list, scores_list, masks_list
                )
        else:
            bboxes, scores, masks = rtmdet_predict(frame, det_model)
            bboxes_list, scores_list, masks_list, frame_list = (
                [bboxes],
                [scores],
                [masks],
                [frame],
            )
    data_samples_list = []

    # for frame, frame_idx, bboxes, scores, masks, poses in zip(
    #     frame_list, frame_idx_list, bboxes_list, scores_list, masks_list, poses_list
    # ):
    for frame, frame_idx, poses in zip(frame_list, frame_idx_list, poses_list):
        # with noneed_timer:
        #     save_masks_json(frame_idx, frame, masks, scores)
        if len(poses) == 0:
            data_samples_list.append(None)
            continue
        data_samples = merge_data_samples(poses)
        bboxes = data_samples.pred_instances.bboxes
        scores = data_samples.pred_instances.bbox_scores
        masks = data_samples.pred_instances.masks

        cropped_images = []
        centroids = []
        mask_areas = []
        for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
            x1, y1, x2, y2 = parse_bbox(bbox)
            cropped_frame = frame[y1:y2, x1:x2]
            cropped_mask = mask[y1:y2, x1:x2]
            mask_areas.append(cropped_mask.sum().item())
            c = find_largest_contour(cropped_mask)
            centroid = compute_centroid(cropped_mask)
            centroid = (centroid[0] + bbox[0], centroid[1] + bbox[1])
            rect = cv2.minAreaRect(c)
            centroids.append(centroid)
            shifted_pose = data_samples.pred_instances.keypoints[i] - np.array([x1, y1])
            with crop_timer:
                img_cropped, rotated_pose = crop_minAreaRect(
                    cropped_frame,
                    rect,
                    shifted_pose,
                    ismask=False,
                )
                mask_cropped, _ = crop_minAreaRect(
                    cropped_mask,
                    rect,
                    shifted_pose,
                    ismask=True,
                )
                img_cropped = np.where(mask_cropped[..., None], img_cropped, 255)

            cropped_images.append(img_cropped)
        # data_samples.pred_instances.masks = masks
        # data_samples.pred_instances.bbox_scores = scores
        data_samples.pred_instances.cropped_images = cropped_images
        data_samples.pred_instances.centroids = centroids
        data_samples.pred_instances.mask_areas = mask_areas
        data_samples_list.append(data_samples)
    if is_batch:
        return data_samples_list
    else:
        return data_samples_list[0]


@numba.jit(nopython=True)
def update_new_mask(mask, width, height, flow, bbox):
    x1, y1, x2, y2 = bbox
    ys, xs = np.where(mask[y1:y2, x1:x2] == 1)
    ys += y1
    xs += x1
    N = len(xs)
    for i in range(N):
        x = xs[i]
        y = ys[i]
        xs[i] = max(min(x + int(round(flow[y, x, 0])), (width - 1)), 0)
        ys[i] = max(min(y + int(round(flow[y, x, 1])), (height - 1)), 0)
    return (ys, xs)


def morph_close(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(src=mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    return mask


def update_masks_by_flow(masks, bboxes, flow):
    N = masks.shape[0]
    # new_masks = np.copy(masks)
    new_masks = np.zeros_like(masks)
    w, h = masks.shape[2], masks.shape[1]
    for i in range(N):
        index_list = update_new_mask(masks[i], w, h, flow, parse_bbox(bboxes[i]))
        new_masks[i][index_list] = 1
        new_masks[i] = morph_close(new_masks[i])
    return new_masks


def get_flow(flow_model, prev_img, img):
    init_default_scope("mmflow")
    scale_factor = 1
    orig_w, orig_h = prev_img[0].shape[1], prev_img[0].shape[0]
    w, h = int(orig_w * scale_factor), int(orig_h * scale_factor)
    prev_img = [cv2.resize(im, (w, h)) for im in prev_img]
    img = [cv2.resize(im, (w, h)) for im in img]
    flow = flow_inference(flow_model, prev_img, img)
    flow = [f.pred_flow_fw.data.permute(1, 2, 0).cpu().numpy() for f in flow]
    flow = [cv2.resize(f, (orig_w, orig_h)) for f in flow]
    return flow


# @numba.jit()
def calc_cost_mat(masks1, bboxes1, masks2, bboxes2):
    bbox_ious = bbox_overlaps(
        torch.tensor(np.array(bboxes1)), torch.tensor(np.array(bboxes2))
    )
    N1 = len(masks1)
    N2 = len(masks2)
    cost_mat = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            if bbox_ious[i, j]:
                x1 = round(min(bboxes1[i][0], bboxes2[j][0]))
                y1 = round(min(bboxes1[i][1], bboxes2[j][1]))
                x2 = round(max(bboxes1[i][2], bboxes2[j][2]))
                y2 = round(max(bboxes1[i][3], bboxes2[j][3]))
                cost_mat[i, j] = iom(
                    masks1[i][y1:y2, x1:x2],
                    masks2[j][y1:y2, x1:x2],
                )
            else:
                cost_mat[i, j] = 0
            # cost_mat[i, j] = iom(masks1[i], masks2[j]) if bbox_ious[i, j] else 0
    return cost_mat


class TrackletManager:
    def __init__(self) -> None:
        self.valid_tracks = []
        self.save_results = []
        self.match_costs = []

    def feed(self, data_samples, frame_idx, flows=None):
        if isinstance(data_samples, (List, Tuple)):
            is_batch = True
            data_samples_list = data_samples
            frame_idx_list = frame_idx
        else:
            is_batch = False
            data_samples_list = [data_samples]
            frame_idx_list = [frame_idx]
        if flows is not None:
            if len(flows) == len(frame_idx_list):
                pass
            elif len(flows) == len(frame_idx_list) - 1:
                flows = [None] + flows
            else:
                raise ValueError("flow length not match frame length")
        else:
            flows = [None] * len(frame_idx_list)
        for data_sample, frame_idx, flow in zip(
            data_samples_list, frame_idx_list, flows
        ):
            # match by mask iou
            if data_sample is None:
                continue
            pred_instances = data_sample.get("pred_instances", None)
            if pred_instances is None:
                continue
            masks_q = [mask for mask in pred_instances.masks]
            centroids_q = pred_instances.centroids
            bboxes_q = pred_instances.bboxes
            bbox_scores = pred_instances.bbox_scores
            ids = [None] * len(pred_instances)

            total_cost = 0
            total_matches = 0
            if frame_idx == 0:
                unmatched_q = list(range(len(pred_instances)))
                costs = np.zeros(len(pred_instances))
            else:
                masks_o = [track.last_mask for track in self.valid_tracks]
                bboxes_o = [track.last_bbox for track in self.valid_tracks]
                if flow is not None:
                    masks_o = update_masks_by_flow(np.array(masks_o), bboxes_o, flow)
                cost_matrix = calc_cost_mat(masks_q, bboxes_q, masks_o, bboxes_o)
                matches, unmatched_q, unmatched_o = match(-cost_matrix)

                costs = np.zeros(len(pred_instances))
                for i, j in matches:
                    costs[i] = round(cost_matrix[i, j], 2)
                    if cost_matrix[i, j] > IOU_THRESHOLD:
                        total_cost += cost_matrix[i, j]
                        total_matches += 1
                        self.valid_tracks[j].update(
                            masks_q[i],
                            centroids_q[i],
                            bboxes_q[i],
                            frame_idx,
                            bbox_scores[i],
                        )
                        ids[i] = self.valid_tracks[j].id
                    else:
                        unmatched_q.append(i)
                    self.match_costs.append(costs[i])

            for q in unmatched_q:
                new_track = Tracklet(
                    frame_idx, masks_q[q], centroids_q[q], bboxes_q[q], bbox_scores[q]
                )
                self.valid_tracks.append(new_track)
                ids[q] = new_track.id
            logging.debug(
                f"{frame_idx}: sum iou {total_cost}, total matches {total_matches}"
            )
            assert None not in ids, "all detections should have an id"
            self.valid_tracks, s = check_valid_tracks(self.valid_tracks, frame_idx)

            for st in s:
                session.add(st.get_stats())
            if len(s):
                session.commit()

            data_sample.pred_instances.instances_id = ids
            data_sample.pred_instances.labels = [
                str(id) + ":" + str(cost) + ":" + str(round(s, 2))
                for id, cost, s in zip(ids, costs, pred_instances.bbox_scores)
            ]
            for cropped_img, id in zip(pred_instances.cropped_images, ids):
                mmcv.imwrite(
                    cropped_img,
                    f"{exp_path}/cropped/{id}/{frame_idx:0>6}.jpg",
                )
        return data_samples_list if is_batch else data_samples_list[0]


if __name__ == "__main__":
    if os.path.exists(f"{exp_path}/cropped"):
        shutil.rmtree(f"{exp_path}/cropped")

    if DET_MODEL == "YOLO":
        from ultralytics import YOLO

        YOLO_WEIGHT = Path(
            "/home/tc/ultralytics/runs/segment/train3/weights/best-seg.pt"
        )
        os.environ["YOLO_VERBOSE"] = "False"
        det_model = YOLO(YOLO_WEIGHT)
    else:
        det_model = init_detector(DET_CONFIG_FILE, DET_CHECKPOINT_FILE, device="cuda:0")
        det_model.cfg.test_dataloader.dataset.pipeline[0].type = "LoadImageFromNDArray"
    # test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)
    if POSE_CONFIG_FILE == "YOLO":
        pose_model = YOLO(POSE_CHECKPOINT_FILE)
    else:
        pose_model = init_pose_estimator(
            POSE_CONFIG_FILE,
            POSE_CHECKPOINT_FILE,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=DRAW_HEATMAP))),
        )
    flow_model = init_flow_estimator(FLOW_CONFIG_FILE, FLOW_CHECKPOINT_FILE)
    video = mmcv.VideoReader(VIDEO_PATH)

    tracklet_manager = TrackletManager()
    frame_idx_list = []
    frame_list = []

    Session = sessionmaker(bind=engine)
    session = Session()
    VIDEO_ID = (
        session.query(Video.id).filter(Video.name == f"{VIDEO_NAME}.mp4").first()[0]
    )
    session.query(Datum).filter(Datum.video_id == VIDEO_ID).delete()
    session.query(TrackletStat).filter(TrackletStat.video_id == VIDEO_ID).delete()
    session.commit()

    prev_frame = None
    scale = 1
    if MAX_WIDTH_HEIGHT > 0:
        width, height = video.width, video.height
        if width > height and width > MAX_WIDTH_HEIGHT:
            scale = MAX_WIDTH_HEIGHT / width
            height = int(height * scale)
            width = MAX_WIDTH_HEIGHT
        elif height > width and height > MAX_WIDTH_HEIGHT:
            scale = MAX_WIDTH_HEIGHT / height
            width = int(width * scale)
            height = MAX_WIDTH_HEIGHT

    for frame_idx, cur_frame in enumerate(track_iter_progress(video)):
        # cur_frame = video[frame_idx + 101925]  # DEBUG
        if frame_idx % 10000 == 10000 - 1:
            gc.collect()
        frame_idx_list.append(frame_idx)
        if MAX_WIDTH_HEIGHT > 0:
            cur_frame = cv2.resize(cur_frame, (width, height))
            frame_list.append(cur_frame)
        else:
            frame_list.append(cur_frame)
        if (
            frame_idx % DET_BATCH_SIZE != DET_BATCH_SIZE - 1
            and frame_idx != len(video) - 1
        ):
            continue

        if ENABLE_FLOW and MAX_DET > 1:
            if prev_frame is None:
                img1s = frame_list[:-1]
                img2s = frame_list[1:]
            else:
                img1s = [prev_frame] + frame_list[:-1]
                img2s = frame_list
            with flow_timer:
                flows = get_flow(flow_model, img1s, img2s)
        else:
            flows = None

        data_samples_list = process_frame(
            frame_list, frame_idx_list, det_model, pose_model
        )
        if MAX_DET > 1:
            data_samples_list = tracklet_manager.feed(
                data_samples_list, frame_idx_list, flows
            )
        else:
            for data_sample in data_samples_list:
                if data_sample is None:
                    continue
                data_sample.pred_instances.instances_id = [1]
                data_sample.pred_instances.labels = ["1"]

        for data_samples, cur_frame, frame_idx in zip(
            data_samples_list, frame_list, frame_idx_list
        ):
            if data_samples is None:
                continue
            pred_instances = data_samples.get("pred_instances", None)

            for (
                id,
                bbox,
                keypoints,
                cropped_img,
                centroid,
                area,
                keypoint_scores,
                bbox_score,
                mask,
            ) in zip(
                pred_instances.instances_id,
                pred_instances.bboxes,
                pred_instances.keypoints,
                pred_instances.cropped_images,
                pred_instances.centroids,
                pred_instances.mask_areas,
                pred_instances.keypoint_scores,
                pred_instances.bbox_scores,
                pred_instances.masks,
            ):
                track_id = 1 if MAX_DET == 1 else None
                if MAX_WIDTH_HEIGHT > 0:
                    bbox = bbox / scale
                    keypoints = keypoints / scale
                    centroid = centroid[0] / scale, centroid[1] / scale
                    area = area / scale / scale
                    mask = cv2.resize(
                        mask,
                        (video.width, video.height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask_rle = maskUtils.encode(
                        np.asfortranarray(mask.astype(np.uint8))
                    )
                datum = Datum(
                    video_id=VIDEO_ID,
                    frame=frame_idx,
                    raw_track_id=id,
                    bbox=str(np.around(bbox, 2).tolist()),
                    bbox_score=bbox_score,
                    keypoints=str(np.around(keypoints, 2).tolist()),
                    keypoint_scores=str(np.around(keypoint_scores, 2).tolist()),
                    centroid=str(centroid),
                    mask_area=area,
                    track_id=track_id,
                    rle=str(maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))),
                )
                session.add(datum)
        session.commit()
        prev_frame = cur_frame
        frame_idx_list = []
        frame_list = []
        if args.first is not None and frame_idx > args.first:
            break

    if MAX_DET > 1:
        for t in tracklet_manager.valid_tracks:
            session.add(t.get_stats())
        if len(tracklet_manager.valid_tracks):
            session.commit()

    del pose_model
    del det_model

    q = session.query(TrackletStat).filter(TrackletStat.video_id == VIDEO_ID)
    logging.info(f"total tracks: {len(q.all())}")
    logging.info(
        f"det: {det_timer.t}, pose: {pose_timer.t}, viz: {viz_timer.t}, crop: {crop_timer.t}, flow: {flow_timer.t}, noneed: {noneed_timer.t}"
    )
