import numpy as np
from typing import Tuple, List
import cv2
import mmcv
from app.algorithm.models import Datum
import re
import logging
from pathlib import Path
import os


def create_logger(name: str, file_path: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if Path(file_path).exists():
        os.remove(file_path)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


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


def enlarge_bboxes(bboxes, maxh, maxw, ratio=0.1):
    larger_bboxes = np.array(bboxes).reshape((-1, 4))
    x1, y1, x2, y2 = (
        larger_bboxes[:, 0],
        larger_bboxes[:, 1],
        larger_bboxes[:, 2],
        larger_bboxes[:, 3],
    )
    w, h = x2 - x1, y2 - y1
    x1 -= (w * ratio).astype(int)
    y1 -= (h * ratio).astype(int)
    x2 += (w * ratio).astype(int)
    y2 += (h * ratio).astype(int)
    larger_bboxes = np.stack((x1, y1, x2, y2), axis=1)
    larger_bboxes[larger_bboxes < 0] = 0
    larger_bboxes[:, 2] = np.clip(larger_bboxes[:, 2], 0, maxw)
    larger_bboxes[:, 3] = np.clip(larger_bboxes[:, 3], 0, maxh)
    return larger_bboxes.astype(int)


def parse_bbox(bbox, h: int, w: int) -> list[int]:
    bbox = np.array(bbox)
    larger_bbox = enlarge_bboxes(bbox, h, w)[0]
    return larger_bbox


def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, 1, 1)
    c = max(contours, key=cv2.contourArea)
    # contours = np.concatenate(contours)
    # c = cv2.convexHull(contours)
    return c


def compute_centroid(mask):
    ys, xs = np.where(mask == 1)
    return np.mean(xs), np.mean(ys)


class InstanceGroup:
    def __init__(
        self,
        scores: np.ndarray,
        bboxes: np.ndarray,
        masks: np.ndarray = None,
        keypoints: np.ndarray = None,
        keypoint_scores: np.ndarray = None,
    ):
        assert len(scores) == len(bboxes), f"Length of scores must match bboxes."
        if masks is not None:
            assert len(scores) == len(masks), f"Length of scores must match masks."
        if keypoints is not None:
            assert len(scores) == len(
                keypoints
            ), f"Length of scores must match keypoints."
        if keypoint_scores is not None:
            assert len(scores) == len(
                keypoint_scores
            ), f"Length of scores must match keypoint_scores."
        self.scores = scores
        self.bboxes = bboxes
        self.masks = masks
        self.keypoints = keypoints
        self.keypoint_scores = keypoint_scores
        self.centroids = []

    def __repr__(self):
        return f"scores: {self.scores}\nbboxes:\n{self.bboxes}\nmasks: {self.masks.shape if self.masks is not None else None}\nkeypoints: {self.keypoints.shape if self.keypoints is not None else None}\nkeypoint_scores: {self.keypoint_scores.shape if self.keypoint_scores is not None else None}"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        assert len(val) == len(self.scores), f"Length of {key} must match scores."
        self.__dict__[key] = val

    def __len__(self):
        return len(self.scores)

    def __copy__(self):
        return InstanceGroup(
            self.scores.copy(),
            self.bboxes.copy(),
            self.masks.copy() if self.masks is not None else None,
            self.keypoints.copy() if self.keypoints is not None else None,
            self.keypoint_scores.copy() if self.keypoint_scores is not None else None,
        )

    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self):
        return self.__copy__()

    def save_images(
        self, frame: np.ndarray, ids: list[int], frame_idx: int, save_dir: str
    ):
        self.centroids: list[tuple] = []
        self.mask_areas: list[int] = []
        for mask, bbox, kpts, id in zip(self.masks, self.bboxes, self.keypoints, ids):
            x1, y1, x2, y2 = parse_bbox(bbox, frame.shape[0], frame.shape[1])
            cropped_frame = frame[y1:y2, x1:x2]
            cropped_mask = mask[y1:y2, x1:x2]

            # TODO: this is ugly :(, need to fix
            centroid = compute_centroid(cropped_mask)
            centroid = (round(centroid[0] + bbox[0]), round(centroid[1] + bbox[1]))
            self.centroids.append(centroid)
            self.mask_areas.append(cropped_mask.sum())

            c = find_largest_contour(cropped_mask)
            rect = cv2.minAreaRect(c)
            shifted_pose = kpts - np.array([x1, y1])
            img_cropped, rotated_pose = crop_min_area_rect(
                cropped_frame, rect, shifted_pose, False
            )
            mask_cropped, _ = crop_min_area_rect(cropped_mask, rect, shifted_pose, True)
            img_cropped = np.where(mask_cropped[..., None], img_cropped, 255)

            img_path = f"{save_dir}/cropped/{id}/{frame_idx:0>6}.jpg"
            mmcv.imwrite(img_cropped, img_path)

    def get_datum(self, video_id, frame_idx, ids, scale=1, track_id=None):
        for mask, bbox, bbox_score, kpts, kpt_scores, centroid, area, id in zip(
            self.masks,
            self.bboxes,
            self.scores,
            self.keypoints,
            self.keypoint_scores,
            self.centroids,
            self.mask_areas,
            ids,
        ):
            if scale != 1:
                bbox = bbox / scale
                kpts = kpts / scale
                centroid = (centroid[0] / scale, centroid[1] / scale)
            bbox_str = str(bbox.astype(int).tolist())
            clean_bbox_str = re.sub(r"\s+", "", bbox_str)

            kpts_str = str(kpts.astype(int).tolist())
            clean_kpts_str = re.sub(r"\s+", "", kpts_str)

            kpt_scores_str = str(np.around(kpt_scores, 2).tolist())
            clean_kpt_scores_str = re.sub(r"\s+", "", kpt_scores_str)

            centroid_str = str(centroid)
            clean_centroid_str = re.sub(r"\s+", "", centroid_str)

            converted_bbox_score = round(bbox_score.item(), 2)
            converted_area = area.item()

            datum = Datum(
                video_id=video_id,
                frame=frame_idx,
                raw_track_id=id,
                bbox=clean_bbox_str,
                bbox_score=converted_bbox_score,
                keypoints=clean_kpts_str,
                keypoint_scores=clean_kpt_scores_str,
                track_id=track_id,
                mask_area=converted_area,
                centroid=clean_centroid_str,
                rle="",
            )
            yield datum


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


def crop_min_area_rect(img, rect, pose, ismask: bool):
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
