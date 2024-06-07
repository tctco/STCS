from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np
import cv2
from mmflow.apis import inference_model
from typing import Tuple
from app.algorithm.models import TrackletStat
import re
from app.algorithm.common import InstanceGroup, match, parse_bbox
import numba
from mmengine.registry import init_default_scope
from mmcv.ops import bbox_overlaps
import torch
from app.algorithm.timer import Profile
import mmcv


class AbstractFlowEstimator(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, images: list[np.ndarray]):
        pass

    def __init__(self, scale=1) -> None:
        self.timer = Profile()
        self.last_image = None
        self.scale = scale


class OpenCVFlowEstimator(AbstractFlowEstimator):
    def __init__(self, scale=1) -> None:
        super().__init__(scale)

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if self.last_image is None:
            self.last_image = mmcv.bgr2gray(images[0])
        flows = []
        with self.timer:
            for i in range(len(images)):
                prev = self.last_image
                curr = mmcv.bgr2gray(images[i])
                flow = cv2.calcOpticalFlowFarneback(
                    prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                flows.append(flow)
                self.last_image = curr
        return flows


class MMFlowEstimator(AbstractFlowEstimator):
    def __init__(self, config: str, checkpoint: str, scale=1, device="cuda:0") -> None:
        super().__init__(scale)
        from mmflow.apis import init_model

        init_default_scope("mmflow")

        self.model = init_model(config, checkpoint, device)

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        init_default_scope(self.model.cfg.get("default_scope", "mmflow"))
        if self.scale != 1:
            h, w = images[0].shape[:2]
            images = [
                cv2.resize(x, (int(w * self.scale), int(h * self.scale)))
                for x in images
            ]
        if self.last_image is None:
            self.last_image = images[0]
        prev_images = [self.last_image] + images[:-1]
        curr_images = images
        with self.timer:
            flow = inference_model(self.model, prev_images, curr_images)
        flow = [x.pred_flow_fw.data.permute(1, 2, 0).cpu().numpy() for x in flow]
        if self.scale != 1:
            flow = [cv2.resize(x, (w, h)) for x in flow]
        self.last_image = images[-1]
        return flow


class AbstractTracker(metaclass=ABCMeta):
    @abstractmethod
    def update(self, image: np.ndarray, dets: Union[InstanceGroup, None]) -> list[int]:
        """Track objects in image.

        Args:
            image (np.ndarray): current image
            dets (Union[InstanceGroup, None]): detections from current frame

        Returns:
            list[int]: list of track ids
        """
        pass

    def __init__(self, max_age: int, iou_threshold: float, w: int, h: int) -> None:
        self.cached_tracklets: list[Tracklet] = []
        self.max_age: int = max_age
        self.w = w
        self.h = h
        self.iou_threshold = iou_threshold
        self.timer = Profile()

    def __repr__(self):
        return self.__class__.__name__


class Tracklet:
    count: int = 0

    def __init__(
        self,
        frame_idx: int,
        mask: np.ndarray,
        centroid: Tuple[float, float],
        bbox: np.ndarray,
        conf: float,
        w: int,
        h: int,
    ) -> None:
        self.id: int = Tracklet.count
        Tracklet.count += 1
        self.h = h
        self.w = w
        self.mask: np.ndarray = mask
        self.bbox: np.ndarray = bbox
        self.start_frame: int = frame_idx
        self.last_frame: int = frame_idx  # included
        self.intervals = []
        self.conf = [conf]
        x1, y1, x2, y2 = parse_bbox(bbox, self.h, self.w)
        self.mask_area = [mask[y1:y2, x1:x2].sum()]
        self.last_centroid: Tuple[float, float] = centroid
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
        self.mask = mask
        self.last_frame = frame_idx
        self.distance += np.linalg.norm(
            np.array(centroid) - np.array(self.last_centroid)
        )
        self.last_centroid = centroid
        self.bbox = bbox
        self.conf.append(conf)
        x1, y1, x2, y2 = parse_bbox(bbox, self.h, self.w)
        self.mask_area.append(mask[y1:y2, x1:x2].sum())

    def get_stats(self, video_id: int):
        intervals_str = str([self.start_frame, *self.intervals, self.last_frame])
        clean_intervals_str = re.sub(r"\s+", "", intervals_str)
        return TrackletStat(
            video_id=video_id,
            track_id=self.id,
            start_frame=self.start_frame,
            end_frame=self.last_frame,
            lifespan=self.last_frame - self.start_frame + 1,
            intervals=clean_intervals_str,
            conf=np.mean(self.conf).item(),
            mask_area=int(np.mean(self.mask_area).item()),
            distance=self.distance,
        )


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
    return ys, xs


def morph_close(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(src=mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    return mask


@numba.jit(nopython=True)
def iom(mask1, mask2):
    return (mask1 & mask2).sum() / max(mask1.sum(), mask2.sum())


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
    return cost_mat


class MaskTracker(AbstractTracker):
    def remove_dead_tracklets(self, frame_idx: int) -> None:
        """remove tracklets that are too old

        Args:
            frame_idx (int): current frame index
        """
        new_cached_tracklets = []
        dead_tracklets = []
        for t in self.cached_tracklets:
            if frame_idx <= t.last_frame + self.max_age:
                new_cached_tracklets.append(t)
            else:
                dead_tracklets.append(t)
        self.cached_tracklets = new_cached_tracklets
        return dead_tracklets

    def update(
        self, dets: Union[InstanceGroup, None], frame_idx: int
    ) -> Tuple[list[int], list[Tracklet]]:
        if dets is None:
            dead_tracklets = self.remove_dead_tracklets(frame_idx)
            return [], dead_tracklets
        ids = [None for _ in range(len(dets))]
        masks_q = dets.masks
        masks_o = [x.mask for x in self.cached_tracklets]
        if len(masks_o) == 0:
            unmatched_q = np.arange(len(dets))
        else:
            bboxes_o = [x.bbox for x in self.cached_tracklets]
            with self.timer:
                cost_matrix = calc_cost_mat(masks_q, dets.bboxes, masks_o, bboxes_o)
            matches, unmatched_q, unmatched_o = match(-cost_matrix)
            for i, j in matches:
                if cost_matrix[i, j] > self.iou_threshold:
                    self.cached_tracklets[j].update(
                        masks_q[i], (0, 0), dets.bboxes[i], frame_idx, dets.scores[i]
                    )
                    ids[i] = self.cached_tracklets[j].id
                else:
                    unmatched_q.append(i)

        for i in unmatched_q:
            new_tracklet = Tracklet(
                frame_idx,
                masks_q[i],
                (0, 0),
                dets.bboxes[i],
                dets.scores[i],
                self.w,
                self.h,
            )
            self.cached_tracklets.append(new_tracklet)
            ids[i] = new_tracklet.id
        assert None not in ids, "all detections should have an id"
        dead_tracklets = self.remove_dead_tracklets(frame_idx)
        return ids, dead_tracklets


class FlowTracker(MaskTracker):
    def update_mask(self, flow: np.ndarray) -> None:
        """update mask and bbox for tracklets

        Args:
            flow (np.ndarray): estimated optical flow
        """
        for t in self.cached_tracklets:
            new_mask = np.zeros_like(t.mask)
            h, w = t.mask.shape[:2]
            index_list = update_new_mask(t.mask, w, h, flow, parse_bbox(t.bbox, h, w))
            new_mask[index_list] = 1
            new_mask = morph_close(new_mask)
            t.mask = new_mask

    def update(
        self, flow: np.ndarray, dets: Union[InstanceGroup, None], frame_idx: int
    ) -> Tuple[list[int], list[Tracklet]]:
        with self.timer:
            self.update_mask(flow)
        return super().update(dets, frame_idx)
