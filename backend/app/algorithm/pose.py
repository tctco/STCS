from mmengine.registry import init_default_scope
from abc import ABCMeta, abstractmethod
from mmengine.dataset import Compose, pseudo_collate
from mmcv.ops import bbox_overlaps
import numpy as np
from typing import Union
import torch
from app.algorithm.common import InstanceGroup, linear_assignment
from app.algorithm.timer import Profile
import mmcv


class AbstractPoseEstimator(metaclass=ABCMeta):
    @abstractmethod
    def predict(
        self, images: list[np.ndarray], dets: list[Union[InstanceGroup, None]]
    ) -> list[Union[InstanceGroup, None]]:
        """Predict pose from images

        Args:
            images (list[np.ndarray]): list of images, must be in BGR!
            dets (list[Union[InstanceGroup, None]]): list of InstanceGroup from detector

        Returns:
            list[Union[InstanceGroup, None]]: list of InstanceGroup with pose information
        """
        pass

    def __call__(
        self, images: list[np.ndarray], dets: list[Union[InstanceGroup, None]]
    ) -> list[Union[InstanceGroup, None]]:
        return self.predict(images, dets)

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def train(self):
        pass

    def __init__(self) -> None:
        self.timer = Profile()


class MMPoseTopDownEstimator(AbstractPoseEstimator):
    def __init__(self, config, checkpoint, device="cuda:0"):
        super().__init__()
        from mmpose.apis import init_model

        self.model = init_model(config, checkpoint, device=device)
        self.pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)
        self.device = device

    def predict(
        self, images: list[np.ndarray], dets: list[Union[InstanceGroup, None]]
    ) -> list[InstanceGroup]:
        assert len(images) == len(
            dets
        ), f"Number of images and InstanceGroups must match. Got {len(images)} images and {len(dets)} InstanceGroups."
        init_default_scope(self.model.cfg.get("default_scope", "mmpose"))
        data_list = []
        result_det_groups = []
        for image, det_group in zip(images, dets):
            if dets is None:
                continue
            for box, score in zip(det_group.bboxes, det_group.scores):
                data_info = dict(
                    img=image, bbox=box[None], bbox_score=np.ones(1) * score
                )
                data_info.update(self.model.dataset_meta)
                data_list.append(self.pipeline(data_info))
        if len(data_list) > 0:
            batch = pseudo_collate(data_list)
            with self.timer:
                with torch.no_grad():
                    result = self.model.test_step(batch)
            lengths = [len(x) for x in dets if x is not None]
            for length, det_group in zip(lengths, dets):
                if det_group is None:
                    result_det_groups.append(None)
                    continue
                kpts = [x.pred_instances.keypoints[0] for x in result[:length]]
                kpt_scores = [x.pred_instances.keypoint_scores for x in result[:length]]
                det_group.keypoints = np.array(kpts)
                det_group.keypoint_scores = np.array(kpt_scores)
                result_det_groups.append(det_group)
                result = result[length:]
        else:
            result_det_groups = [None] * len(dets)
        return result_det_groups

    def train(self):
        pass


class YOLOPoseBottomUpEstimator(AbstractPoseEstimator):
    def __init__(self, checkpoint: str, device: str = "cuda:0"):
        super().__init__()
        from ultralytics import YOLO

        self.model = YOLO(checkpoint)

    def train(self):
        pass

    def predict(
        self, images: list[np.ndarray], dets: list[Union[InstanceGroup, None]]
    ) -> list[Union[InstanceGroup, None]]:
        """Predict pose from images

        Args:
            images (list[np.ndarray]): list of images, must be in RGB

        Returns:
            list[InstanceGroup]: list of InstanceGroup
        """
        images = [mmcv.bgr2rgb(x) for x in images]
        with self.timer:
            results = self.model(images, verbose=False)
        instance_groups = []
        for result, det_group in zip(results, dets):
            if det_group is None:
                continue
            pred_bboxes = result.boxes.data.cpu().numpy()[:, :4]
            if len(pred_bboxes) == 0:
                continue

            iou_mat = bbox_overlaps(
                torch.tensor(det_group.bboxes), torch.tensor(pred_bboxes)
            ).numpy()
            matches = linear_assignment(-iou_mat)
            scores, bboxes, masks, keypoints, keypoint_scores = [], [], [], [], []
            pred_kpts = result.keypoints.data.cpu().numpy()
            for i, j in matches:
                scores.append(det_group.scores[i])
                bboxes.append(det_group.bboxes[i])
                masks.append(det_group.masks[i])
                keypoints.append(pred_kpts[j, :, :2])
                keypoint_scores.append(pred_kpts[j, :, 2])
            instance_groups.append(
                InstanceGroup(
                    np.array(scores),
                    np.array(bboxes),
                    np.array(masks),
                    np.array(keypoints),
                    np.array(keypoint_scores),
                )
            )
        return instance_groups


if __name__ == "__main__":
    DET_CONFIG_FILE = "./trained_models/2023-07-12_mouse_det/rtmdet-ins_tiny.py"
    DET_CHECKPOINT_FILE = (
        "./trained_models/2023-07-12_mouse_det/best_coco_segm_mAP_epoch_155.pth"
    )
    POSE_CHECKPOINT_FILE = "./trained_models/yolov8pose/weights/best.pt"
    import mmcv

    video = mmcv.VideoReader(
        "/mnt/e/expresults/segTracker_supp_materials/original_videos/6xWT.mp4"
    )
    from det import MMDetImageDetector

    det_model = MMDetImageDetector(DET_CONFIG_FILE, DET_CHECKPOINT_FILE)
    pose_model = YOLOPoseBottomUpEstimator(POSE_CHECKPOINT_FILE)

    frame = video[0]
    det_group = det_model([frame])
    pose_group = pose_model([mmcv.bgr2rgb(frame)], det_group.copy())
    print(pose_group)

    POSE_CONFIG_FILE = "./trained_models/2023-06-10_mouse_pose/hm_mobilenetv2.py"
    POSE_CHECKPOINT_FILE = (
        "./trained_models/2023-06-10_mouse_pose/best_coco_AP_epoch_86.pth"
    )
    pose_model = MMPoseTopDownEstimator(POSE_CONFIG_FILE, POSE_CHECKPOINT_FILE)
    pose_group = pose_model([frame], det_group)
    print(pose_group)
