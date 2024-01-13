from app.api.videos.models import Video
from app.algorithm.models import Datum
import numpy as np
from sqlalchemy import func as sql_func
import ast
import pandas as pd


class SocialActionCluster:
    def __init__(self, session, config, num_kpts=8) -> None:
        self.session = session
        self.num_kpts = 8

    def _export_keypoints(self, vid: int):
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

    def generate_data(self, video_ids, time_window, down_sample_rate=2, targets=None):
        """
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
        if targets is None:
            for vid in video_ids:
                keypoints, centroids = self._export_keypoints(vid)

        else:
            raise NotImplementedError("Not implemented yet")
