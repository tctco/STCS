from mmengine.registry import init_default_scope
from abc import ABCMeta, abstractmethod
import numpy as np
from mmdet.utils import get_test_pipeline_cfg
from mmengine.dataset import Compose, pseudo_collate
import torch
from mmcv.ops import soft_nms
from app.algorithm.timer import Profile
from app.algorithm.common import InstanceGroup
from typing import Union
import mmcv
import cv2


class AbstractDetector(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, max_det: int, conf_threshold: float, nms_threshold: float):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_det: int = max_det

    @abstractmethod
    def predict(self, images: list[np.ndarray]) -> list[Union[InstanceGroup, None]]:
        """Detect objects in images.

        Args:
            images (list[np.ndarray]): List of images, must be in BGR!

        Returns:
            list[InstanceGroup]: List of instances
        """
        pass

    def __call__(self, images: list[np.ndarray]) -> list[Union[InstanceGroup, None]]:
        return self.predict(images)

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def train(self):
        pass

    def process_det_pred_instances(
        self, pred_instances
    ) -> Union[list[InstanceGroup], None]:
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        masks = pred_instances.masks.cpu().numpy().astype(np.uint8)
        # bboxes = np.concatenate(
        # (bboxes, scores[:, None]), axis=1)

        _, nms_result = soft_nms(bboxes, scores, self.nms_threshold)
        # _, nms_result = batched_nms(bboxes, scores, {'iou_threshold':NMS_THRESH})
        bboxes = bboxes[nms_result]
        masks = masks[nms_result]
        scores = scores[nms_result]

        scores_result = scores > self.conf_threshold
        bboxes = bboxes[scores_result]
        masks = masks[scores_result]
        scores = scores[scores_result]
        if len(bboxes) > self.max_det:
            bboxes = bboxes[: self.max_det]
            masks = masks[: self.max_det]
            scores = scores[: self.max_det]
        # assert len(bboxes) == len(masks) and len(masks) == len(scores)
        if len(scores) == 0:
            return None
        return InstanceGroup(scores, bboxes, masks)


class MMDetImageDetector(AbstractDetector):
    def __init__(
        self,
        max_det: int,
        conf_threshold: float,
        nms_threshold: float,
        config_file: str,
        checkpoint_file: str,
        device="cuda:0",
    ):
        super().__init__(max_det, conf_threshold, nms_threshold)
        from mmdet.apis import init_detector

        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.timer = Profile()

    def predict(self, images: list[np.ndarray]) -> list[Union[InstanceGroup, None]]:
        init_default_scope(self.model.cfg.get("default_scope", "mmdet"))
        cfg = self.model.cfg
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        test_pipeline[0].type = "mmdet.LoadImageFromNDArray"
        test_pipeline = Compose(test_pipeline)
        data = [{"img": x, "img_id": i} for i, x in enumerate(images)]
        data = [test_pipeline(x) for x in data]
        data[0]["inputs"] = [x["inputs"] for x in data]
        data[0]["data_samples"] = [x["data_samples"] for x in data]
        detections = []
        with self.timer:
            with torch.no_grad():
                results = self.model.test_step(data[0])
        for result in results:
            pred_instances = result.get("pred_instances", None)
            if pred_instances is None:
                detections.append(None)
            else:
                detections.append(self.process_det_pred_instances(pred_instances))
        return detections

    def train(self):
        pass


class YOLODetector(AbstractDetector):
    def __init__(
        self,
        max_det: int,
        conf_threshold: float,
        nms_threshold: float,
        checkpoint_file: str,
        device="cuda:0",
    ):
        super().__init__(max_det, conf_threshold, nms_threshold)
        from ultralytics import YOLO

        self.model = YOLO(checkpoint_file)
        self.timer = Profile()

    def predict(self, images: list[np.ndarray]) -> list[Union[InstanceGroup, None]]:
        images = [mmcv.bgr2rgb(x) for x in images]
        results = self.model(
            images,
            max_det=self.max_det,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            verbose=False,
        )
        detections = []
        y, x, c = images[0].shape
        for result in results:
            if result.boxes is None or result.masks is None:
                detections.append(None)
                continue
            bboxes = result.boxes.data.cpu().numpy()
            bboxes, scores = bboxes[:, :4], bboxes[:, 4]
            masks = result.masks.data.cpu().numpy().astype(np.uint8)
            masks = [cv2.resize(m, (x, y)) for m in masks]
            masks = np.stack(masks)
            detections.append(InstanceGroup(scores, bboxes, masks))
        return detections

    def train(self):
        pass
