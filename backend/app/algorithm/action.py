import pickle
from pathlib import Path
from app.api.videos.models import Video
from app.algorithm.models import Datum
import numpy as np
from sqlalchemy import func as sql_func
import ast
import pandas as pd
from mmengine import Config
from mmengine.runner import Runner
from mmaction.apis import init_recognizer
from mmengine.dataset import Compose, pseudo_collate
import torch


class SocialActionCluster:
    def __init__(
        self,
        session,
        config,
        save_path: Path,
        jid,
        video_ids,
        labels,
        time_window,
        downsample_rate,
    ) -> None:
        self.session = session
        self.num_kpts = 8
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        self.save_path = save_path
        self.pickle_path: Path = self.generate_data(
            jid, video_ids, labels, time_window, downsample_rate
        )
        self.config = self._build_config(config, str(self.pickle_path))

    def _build_config(self, config_path: str, data_path: str):
        config = Config.fromfile(config_path)
        config.train_dataloader.dataset.dataset["ann_file"] = data_path
        config.val_dataloader.dataset["ann_file"] = data_path
        config.test_dataloader.dataset["ann_file"] = data_path
        config.work_dir = str(self.save_path)
        return config

    def train(self):
        runner = Runner.from_cfg(self.config)
        runner.train()

    def feature(self, checkpoint_path: str):
        batch_size = 16
        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)
        val_titles = data["split"]["xub_val"]
        val_data = []
        for item in data["annotations"]:
            if item["frame_dir"] in val_titles:
                val_data.append(item)
        model = init_recognizer(self.config, checkpoint_path)
        model.eval()
        test_pipeline = Compose(self.config.test_pipeline)

        features = []
        for batch in range(0, len(val_data), batch_size):
            batch_data = val_data[batch : batch + batch_size]
            batch_data = [test_pipeline(x) for x in batch_data]
            batch_data = pseudo_collate(batch_data)
            with torch.no_grad():
                processed = model.data_preprocessor(batch_data, False)
                res = model(
                    processed["inputs"], processed["data_samples"], mode="predict"
                )
                features.extend([x.feat.item.cpu().numpy() for x in res])
        return features

    def _export_keypoints(self, vid: int) -> tuple[np.ndarray, np.ndarray]:
        video = self.session.query(Video).filter(Video.id == vid).first()
        assert (
            video is not None and video.analyzed
        ), f"Video {vid} not found or not tracked (tracked? {video.analyzed})"
        max_frame = video.frame_cnt
        data = self.session.query(Datum).filter(Datum.video_id == vid)
        max_det = (
            self.session.query(sql_func.max(Datum.det_id))
            .filter(Datum.video_id == vid)
            .scalar()
        )
        keypoints = np.zeros((max_frame, max_det, 8, 3))
        centroids = np.empty((len(video), max_det, 2))
        centroids[:] = np.nan
        for d in data:
            keypoints[d.frame, d.track_id - 1] = ast.literal_eval(d.keypoints)
            centroids[d.frame, d.track_id - 1] = ast.literal_eval(d.centroid)
        for i in range(max_det):
            _df = pd.DataFrame(centroids[:, i, :])
            _df = _df.interpolate()
            centroids[:, i, :] = _df.values
        return keypoints, centroids

    def generate_data(
        self, jid, video_ids, labels, time_window, down_sample_rate=2, targets=None
    ) -> Path:
        """
        https://mmaction2.readthedocs.io/en/latest/dataset_zoo/skeleton.html#the-format-of-annotations
        {
          'split':{'xub_train':[], 'xub_val':[]},
          'annotations': [{
            'keypoint': np.array[individuals x frames x kpts x 2],
            'label': animal genotype index,
            'keypoint_score': np.array[individuals x frames x kpts],
            'total_frames': int,
            'img_shape': (int, int),
            'original_shape': same as img_shape,
            'frame_dir': xsub_train/val name,
            'comb': animal combination (int, int, ...),
            'video_id': int,
            'target': index of target animal, could be None if all animals are the same
          }]
        }
        """
        time_window = time_window // down_sample_rate
        result = {"split": {"xub_train": [], "xub_val": []}, "annotations": []}
        if targets is not None:
            raise NotImplementedError("Not implemented yet")

        for vid, label in zip(video_ids, labels):
            keypoints, centroids = self._export_keypoints(vid)
            video = self.session.query(Video).filter(Video.id == vid).first()
            keypoints = keypoints[::down_sample_rate]
            centroids = centroids[::down_sample_rate]
            n_windows = round(len(keypoints) / time_window)
            splited_keypoints = np.array_split(keypoints, n_windows)
            shifted_keypoints = keypoints[time_window // 2 :]
            shifted_n_windows = round(len(shifted_keypoints) / time_window)
            shifted_splited_keypoints = np.array_split(
                shifted_keypoints, shifted_n_windows
            )
            train = (
                splited_keypoints[: n_windows // 2]
                + shifted_splited_keypoints[: shifted_n_windows // 2]
            )
            val = (
                splited_keypoints[n_windows // 2 :]
                + shifted_splited_keypoints[shifted_n_windows // 2 :]
            )
            xub_train, annotation_train = self._parse_chunks(video, train, label)
            xub_val, annotation_val = self._parse_chunks(video, val, label)
            result["split"]["xub_train"].extend(xub_train)
            result["split"]["xub_val"].extend(xub_val)
            result["annotations"].extend(annotation_train)
            result["annotations"].extend(annotation_val)
        with open(self.save_path / f"{jid}.pkl", "wb") as f:
            pickle.dump(result, f)
        return self.save_path / f"{jid}.pkl"

    def _parse_chunks(self, video: Video, chunks: list[np.ndarray], label: int):
        xub, annotation = [], []
        for i, arr in enumerate(chunks):
            arr = arr.transpose(1, 0, 2, 3)
            xub.append(f"{video.id}/{i}")
            annotation.append(
                {
                    "keypoint": arr[..., :2],
                    "label": label,
                    "keypoint_score": arr[..., 2],
                    "total_frames": arr.shape[1],
                    "img_shape": (video.width, video.height),
                    "original_shape": (video.width, video.height),
                    "frame_dir": f"{video.id}/{i}",
                    "comb": list(range(len(arr))),
                    "video_id": video.id,
                    # "target": None,
                }
            )
        return xub, annotation
