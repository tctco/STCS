from typing import List, Tuple, Union
import os
import logging
from pathlib import Path
import datetime
from constants import (
    VIDEO_NAME,
    CLS_CONFIG_FILE,
    VIDEO_PATH,
    MIN_MERGE_FRAMES,
    APPEARANCE_THRESHOLD,
    BATCH_SIZE,
    MAX_STRIDE,
    MIN_STRIDE,
    INFERENCE_STRIDE,
    ACCUMULATED_FRAME_RATIO,
    MAX_DET,
    SOFT_BORDER,
    MAX_CROP_SIZE,
    MIN_CLS_CONF,
)
import sys
import os
import pickle
from models import TrackletStat, Datum
from app.api.videos.models import Video
from app.common.database import engine
from sqlalchemy.orm import sessionmaker
import ast

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("video_name", help="video name without extension", type=str)
parser.add_argument("video_path", help="video root path", type=str)
parser.add_argument("max_det", help="maximum animals in the video", type=int)
parser.add_argument("--resume", help="resume from checkpoint", type=int)
args = parser.parse_args()
VIDEO_NAME = args.video_name
VIDEO_PATH = f"{args.video_path}/{VIDEO_NAME}.mp4"
MAX_DET = args.max_det
session = sessionmaker(bind=engine)()
VIDEO_ID = session.query(Video.id).filter(Video.name == f"{VIDEO_NAME}.mp4").first()[0]


if args.resume is not None:
    resume = args.resume
else:
    resume = -1
str_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
Path("./exp_results").mkdir(parents=True, exist_ok=True)
logging.getLogger("fontTools").setLevel("WARNING")
logging.getLogger("matplotlib").setLevel("WARNING")
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("torch").setLevel(logging.INFO)
logger = logging.getLogger("track")
logging.basicConfig(
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(f"./exp_logs/{VIDEO_NAME}_track_{str_time}.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

import numpy as np
import shutil
from mmcls import init_model
import json
from mmengine import Config
from mmengine.runner import Runner
import os.path as osp
import logging
from mmengine.utils import track_iter_progress
import random
import torch
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import pickle
import mmcv
from .timer import Profile

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

cls_timer = Profile()
train_timer = Profile()
group_track_timer = Profile()

SCALE_FACTOR = 1


def random_color(seed):
    """Random a color according to the input seed."""
    np.random.seed(seed)
    color = (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
    )
    return color


class IntervalOverlapError(Exception):
    def __init__(self, track1: "Track", track2: "Track") -> None:
        self.track1: "Track" = track1
        self.track2: "Track" = track2

    def __str__(self):
        return (
            f"intervals should not overlap:"
            f"\n{self.track1.id} {self.track1.intervals} {self.track1.merged}\n"
            f"{self.track2.id} {self.track2.intervals} {self.track2.merged}"
        )


class ConflictAssignedClassError(Exception):
    def __init__(self, track1: "Track", track2: "Track") -> None:
        self.t1 = track1
        self.t2 = track2

    def __str__(self) -> str:
        return f"Assigned classes should not conflict: {self.t1.id}: {self.t1.assigned_class} {self.t2.id}: {self.t2.assigned_class}"


class UnreliableClassificationModelError(Exception):
    def __init__(self, top1acc: float) -> None:
        self.top1acc = top1acc

    def __str__(self) -> str:
        return (
            f"Classification model is unreliable: The validation acc is only {self.top1acc}. "
            "Probably because there are mixed tracks."
        )


class InvalidCliqueError(Exception):
    def __init__(self, clique) -> None:
        self.clique = clique

    def __str__(self):
        return f"Clique size {self.clique} larger than maxdet"


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def is_overlap(
    intervals1: List[Tuple[int, int]],
    intervals2: List[Tuple[int, int]],
    threshold: int = SOFT_BORDER,
):
    i, j = 0, 0
    while i < len(intervals1) and j < len(intervals2):
        if intervals1[i][1] < intervals2[j][0]:
            i += 1
        elif intervals2[j][1] < intervals1[i][0]:
            j += 1
        else:
            tmp = sorted(intervals1[i] + intervals2[j])
            if tmp[2] - tmp[1] >= threshold:
                return True
            else:
                if intervals1[i][1] < intervals2[j][1]:
                    i += 1
                else:
                    j += 1
    return False


def frame_cnt(intervals: List[Tuple[int, int]]):
    cnt = 0
    for i in intervals:
        cnt += i[1] - i[0] + 1
    return cnt


class Track:
    cnt = 0

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        intervals=None,
        id=None,
        bbox_conf=1,
        distance=-1,
    ) -> None:
        assert start_frame <= end_frame, "start_frame should be less than end_frame"
        if id is None:
            self.id = Track.cnt
        else:
            self.id = id
        Track.cnt += 1
        if intervals is None:
            self.intervals: List[Tuple[int, int]] = [(start_frame, end_frame)]
        else:
            self.intervals = intervals
        self.merged: set = set([self.id])
        self.frame_cnt: int = frame_cnt(self.intervals)
        self.assigned_class: Union[None, int] = None
        self.pred = None
        self.original_id = self.id
        self.conf = bbox_conf
        self.triggered_contradictions = 0
        self.speed = distance / self.frame_cnt

    def __repr__(self) -> str:
        return f"{self.id}: framecnt: {self.frame_cnt}, merged: {sorted(self.merged)}, conf: {self.conf}, cls: {self.assigned_class}, intervals: {self.intervals}"

    def merge(self, track: "Track", ignore_overlap=False):
        logger.debug(
            f"merging {self.id}({self.original_id}) and {track.id}({track.original_id})"
        )
        if not ignore_overlap and is_overlap(track.intervals, self.intervals):
            raise IntervalOverlapError(self, track)

        if self.assigned_class is not None and track.assigned_class is not None:
            if self.assigned_class != track.assigned_class:
                raise ConflictAssignedClassError(self, track)
        if self.assigned_class is None:
            self.assigned_class = track.assigned_class
        self.intervals.extend(track.intervals)
        self.intervals.sort(key=lambda x: x[0])
        self.merged |= track.merged
        boolean_array = np.zeros(
            max(self.intervals, key=lambda x: x[1])[1] + 1, dtype=bool
        )
        for i in self.intervals:
            boolean_array[i[0] : i[1] + 1] = True
        padded = np.hstack([[False], boolean_array, [False]])  # padding
        d = np.diff(padded.astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1

        new_interval = list(zip(starts, ends))
        self.intervals = new_interval
        if self.pred is None or track.pred is None:
            self.pred = None
        else:
            self.pred = (self.pred * self.frame_cnt + track.pred * track.frame_cnt) / (
                self.frame_cnt + track.frame_cnt
            )
        self.frame_cnt = frame_cnt(self.intervals)


class TrackMerger:
    def __init__(self, tracks: List[Track], max_det) -> None:
        self.max_det: int = max_det
        self.tracks: List[Track] = tracks
        logger.info(f"total tracks: {len(self.tracks)}")
        self._original_tracks = [t for t in tracks]
        self.track_groups = []  # 考虑用优先队列
        self.contradictory_tracks = []
        self._trash: List[int] = []
        self.cls_model = None
        self.cls_model_updated = False
        self.dataset_exported = False
        self.plot_cnt = 0
        self.plot_network_cnt = 0
        self.merged = {}
        self.trained_times = [0] * len(tracks)
        self.backup_cnt = 0
        self.first_train = True
        self.dumped: List[Track] = []
        self.saved_state = False
        pics_root = Path(f"./exp_files/{VIDEO_NAME}/cropped")
        img_shapes = []
        self.filter_tracks(min_conf=MIN_CLS_CONF)
        for t in self.tracks:
            for fname in os.listdir(pics_root / str(t.id))[::5]:
                fpath = pics_root / str(t.id) / fname
                img = cv2.imread(str(fpath))
                img_shapes.append(img.shape[:2])
        img_shapes = np.array(img_shapes)
        img_shape = np.percentile(img_shapes, 95, axis=0) // 32 * 32 + 32
        if max(img_shape) > MAX_CROP_SIZE:
            self.scale_factor = MAX_CROP_SIZE / max(img_shape)
            self.img_shape = int(img_shape[0] * self.scale_factor), int(
                img_shape[1] * self.scale_factor
            )
        else:
            self.scale_factor = 1
            self.img_shape = int(img_shape[0]), int(img_shape[1])
        logger.info(f"computed img shape (h x w): {self.img_shape}")

    def build_cfg(self, config_file_path):
        cfg = Config.fromfile(config_file_path)
        work_dir = osp.join(
            f"./exp_cls_models/cls_{VIDEO_NAME}_work_dirs",
            osp.splitext(osp.basename(CLS_CONFIG_FILE))[0],
        )
        cfg.work_dir = work_dir
        # cfg.train_dataloader.dataset.dataset.data_root = f'/home/tc/open-mmlab/tracker/{VIDEO_NAME}/cropped/'
        cfg.train_dataloader.dataset.data_root = f"./exp_files/{VIDEO_NAME}/cropped/"
        cfg.val_dataloader.dataset.data_root = f"./exp_files/{VIDEO_NAME}/cropped/"
        cfg.test_dataloader.dataset.data_root = f"./exp_files/{VIDEO_NAME}/cropped/"
        cfg.train_dataloader.batch_size = BATCH_SIZE
        cfg.val_dataloader.batch_size = BATCH_SIZE
        cfg.test_dataloader.batch_size = BATCH_SIZE
        cfg.auto_scale_lr.base_batch_size = BATCH_SIZE
        cfg.train_dataloader.dataset.pipeline[1]["crop_size"] = self.img_shape
        cfg.val_dataloader.dataset.pipeline[1]["crop_size"] = self.img_shape
        cfg.test_dataloader.dataset.pipeline[1]["crop_size"] = self.img_shape
        if self.scale_factor < 1:
            cfg.train_dataloader.dataset.pipeline.insert(
                1, dict(type="Resize", scale_factor=self.scale_factor)
            )
            cfg.val_dataloader.dataset.pipeline.insert(
                1, dict(type="Resize", scale_factor=self.scale_factor)
            )
            cfg.test_dataloader.dataset.pipeline.insert(
                1, dict(type="Resize", scale_factor=self.scale_factor)
            )

        cfg.model.head["num_classes"] = self.max_det
        cfg.data_preprocessor["num_classes"] = self.max_det
        return cfg

    def update_cls_model(self, total_train=1000) -> dict:
        """
        return: {'accuracy/top1': 37.45582962036133, 'accuracy/top2': 41.69611358642578}
        """
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        cfg = self.build_cfg(CLS_CONFIG_FILE)

        best_cls_model_dir = Path(
            f"./exp_cls_models/cls_{VIDEO_NAME}_work_dirs/osnet_x0_25"
        )
        best_model = None
        if (best_cls_model_dir / "best_accuracy.pth").exists():
            cfg.load_from = f"exp_cls_models/cls_{VIDEO_NAME}_work_dirs/osnet_x0_25/best_accuracy.pth"
        else:
            cfg.load_from = "./trained_models/osnet/osnet_x0_25_imagenet_renamed.pth"

        cfg.model.head["num_classes"] = self.max_det
        cfg.data_preprocessor["num_classes"] = self.max_det

        if self.first_train:
            self.first_train = False
            cfg.custom_hooks[0]["patience"] = 12
        elif total_train > 1800 * self.max_det:
            cfg.custom_hooks[0]["patience"] = 3 * round(
                1 + self.new_imgs_cnt * 5 / self.old_imgs_cnt
            )  # TODO: DEBUG
        else:
            cfg.custom_hooks[0]["patience"] = 5 * round(
                1 + self.new_imgs_cnt * 2.5 / self.old_imgs_cnt
            )
        cls_runner = Runner.from_cfg(cfg)
        with train_timer:
            cls_runner.train()
        self.cls_model_updated = True

        for model in os.listdir(best_cls_model_dir):
            if model.endswith(".pth") and "best" in model and "epoch" in model:
                best_model = model
                break
        if best_model:
            best_cls_path = best_cls_model_dir / best_model
            shutil.move(best_cls_path, best_cls_model_dir / "best_accuracy.pth")

        return cls_runner.val(), str(best_cls_model_dir / "best_accuracy.pth")

    def merge(self, track_ids: set):
        track_ids = sorted(list(track_ids), reverse=True)
        for t in track_ids[:-1]:
            try:
                self.tracks[track_ids[-1]].merge(self.tracks[t])
                self._trash.append(t)
            except IntervalOverlapError as e:
                logger.error(e)
                logger.warning(
                    f"in order to keep the program running, we will remove track {t}"
                )
                if (
                    self.tracks[t].frame_cnt <= self.tracks[track_ids[-1]].frame_cnt
                    and self.tracks[t].frame_cnt <= 100
                ):
                    self._trash.append(t)
                else:
                    self.tracks[track_ids[-1]].merge(
                        self.tracks[t], ignore_overlap=True
                    )
                    self._trash.append(t)

    def _clean(self, action=""):
        # TODO 优化
        if action == "filter":
            for t in self._trash:
                for m in self.tracks[t].merged:
                    self.dumped.append(self._original_tracks[m])
        trash = sorted(self._trash, reverse=True)
        for i, t in enumerate(trash):
            for track in self.tracks[t + 1 :]:
                track.id -= 1
            poped = self.tracks.pop(t)
            poped.action = action
            self.merged[poped.original_id] = poped
        Track.cnt -= len(self._trash)
        self._trash = []

    def filter_tracks(self, min_len=10, min_conf=0.6, export_difficult_frames=True):
        logger.info("filtering tracks...")
        # data_root = Path(f"./exp_files/{VIDEO_NAME}/cropped/")
        # mean_areas = np.zeros(len(self.tracks))
        # areas = []
        # for id in track_iter_progress(os.listdir(data_root)):
        #     if id.endswith(".json"):
        #         continue
        #     mean_area = []
        #     for fname in os.listdir(data_root / str(id)):
        #         img = cv2.imread(str(data_root / str(id) / fname))
        #         area = (img < 255).all(axis=2).sum()
        #         mean_area.append(area)
        #         areas.append(area)
        #     mean_areas[int(id)] = np.array(mean_area).mean()
        # areas = np.array(areas)
        # area_threshold = np.percentile(areas, 90)
        # # TODO: also add lower bound?
        areas = session.query(Datum.mask_area).filter(Datum.video_id == VIDEO_ID).all()
        mean_areas = np.array(
            session.query(TrackletStat.mask_area)
            .filter(TrackletStat.video_id == VIDEO_ID)
            .order_by(TrackletStat.track_id)
            .all()
        ).ravel()
        area_threshold = np.mean(areas) + 2 * np.std(areas)
        too_large_area = list(np.arange(len(self.tracks))[mean_areas > area_threshold])
        logger.info(
            f"{len(too_large_area)} tracks with too large mean area ({too_large_area})"
        )
        speeds = []
        too_short, not_confident = [], []
        for t in self.tracks:
            speeds.append(t.speed)
            if t.frame_cnt <= min_len:
                too_short.append(t.id)
            elif t.conf < min_conf:
                not_confident.append(t.id)
        speed_threshold = np.mean(speeds) - 2 * np.std(speeds)
        too_slow = [t.id for t in self.tracks if t.speed < speed_threshold]

        logger.info(
            f"{len(too_short)} tracks are too short, {len(not_confident)} tracks are not confident, {len(too_slow)} tracks are too slow, {len(too_large_area)} tracks have too large mean area"
        )
        self._trash = list(set(too_short + not_confident + too_large_area + too_slow))
        if export_difficult_frames:
            self.export_difficult_frames(too_large_area + not_confident)
        self._clean("filter")

    def export_difficult_frames(self, original_track_ids):
        video = mmcv.VideoReader(VIDEO_PATH)
        difficult_frames = []
        target_dir = Path(f"./exp_files/{VIDEO_NAME}/difficult_frames/")
        logger.info(
            f"Exporting difficult frames... You may need to annotate these frames for better performance. The images are saved to ./exp_files/{VIDEO_NAME}/difficult_frames/"
        )
        if osp.exists(target_dir):
            shutil.rmtree(target_dir)
        target_dir.mkdir()
        img_root = Path(f"./exp_files/{VIDEO_NAME}/cropped/")
        for id_str in os.listdir(img_root):
            if id_str.endswith(".json"):
                continue
            id = int(id_str)
            if id not in original_track_ids:
                continue
            for fname in os.listdir(img_root / id_str):
                frame = int(fname.split(".")[0])
                difficult_frames.append(frame)
        difficult_frames.sort()
        last_frame = -float("inf")
        for i in range(len(difficult_frames) - 1):
            if i - last_frame > video.fps * 4:
                mmcv.imwrite(
                    video[difficult_frames[i]],
                    target_dir
                    / f"{VIDEO_NAME}_{difficult_frames[i]:0>8}_difficult.jpg",
                )
                last_frame = i

    def update_tracks_groups(self, discard=False):
        G = nx.Graph()
        for a, s in enumerate(self.contradictory_tracks):
            for b in s:
                if a == b:
                    continue
                G.add_edge(a, b)
            if s == {a}:
                G.add_node(a)
        with group_track_timer:
            cliques = nx.find_cliques(G)
        logger.debug(f"Finding cliques takes {group_track_timer.dt}s")
        self._draw_network(G)
        result = []
        for c in cliques:
            if len(c) > self.max_det:
                if not discard:
                    logger.error(f"{[self.tracks[id] for id in c]}")
                if discard:
                    c = sorted(c, key=lambda x: self.tracks[x].frame_cnt, reverse=True)
                    self._trash.extend(c[self.max_det :])
                    logger.warning(
                        f"discard short tracks {c[self.max_det:]}, lengths are {[self.tracks[x].frame_cnt for x in c[self.max_det:]]}"
                    )
                    c = c[: self.max_det]
                else:
                    raise InvalidCliqueError(c)
            lifespan = min([self.tracks[id].frame_cnt for id in c])
            result.append({"ids": c, "lifespan": lifespan})
        self.track_groups = sorted(
            result, key=lambda x: (len(x["ids"]), x["lifespan"]), reverse=True
        )
        if discard:
            self._trash = list(set(self._trash))
            if len(self._trash):
                logger.debug(f"need to clean {sorted(self._trash)}")
                self._clean("clique")
                logger.debug(
                    f"after clique clean: {[(t.id, t.original_id, t.frame_cnt, t.assigned_class) for t in self.tracks]}"
                )
                return False
            else:
                return True
        return True

    def _draw_network(self, G: nx.Graph):
        plt.close("all")
        return
        colors = []
        sizes = []
        alphas = []
        fixed = [-1] * self.max_det
        default_pos = {}
        for n in G.nodes:
            sizes.append(self.tracks[n].frame_cnt ** 0.6)
            if self.tracks[n].assigned_class is not None:
                color = random_color(self.tracks[n].assigned_class)
                color = [c / 255 for c in color]
                colors.append(color)
                fixed[self.tracks[n].assigned_class] = n
                alphas.append(0.8)
            else:
                colors.append("gray")
                alphas.append(0.4)
        if -1 in fixed:
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(
                G, pos, node_color=colors, node_size=sizes, alpha=alphas
            )
            nx.draw_networkx_edges(G, pos, alpha=0.2)
            nx.draw_networkx_labels(G, pos, font_size=8, alpha=0.8)
            plt.savefig(
                f"./{VIDEO_NAME}/plots/network_{self.plot_network_cnt}.pdf",
                format="pdf",
                bbox_inches="tight",
            )
            self.plot_network_cnt += 1
            return
        for i, n in enumerate(fixed):
            theta = np.pi * 2 / self.max_det * i
            default_pos[n] = (1 * np.cos(theta), 1 * np.sin(theta))
        pos = nx.spring_layout(G, fixed=fixed, pos=default_pos, center=(0, 0), scale=1)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=alphas)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos, font_size=8, alpha=0.8)
        plt.savefig(
            f"./{VIDEO_NAME}/plots/network_{self.plot_network_cnt}.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        self.plot_network_cnt += 1

    def update_contradictory_tracks(self, threshold=SOFT_BORDER):
        logger.info(f"updating contradictory tracks with threshold {threshold}...")
        result = [set() for _ in range(len(self.tracks))]
        tracks = self.tracks.copy()
        tracks.sort(key=lambda x: x.intervals[-1][1])
        for i, t1 in enumerate(tracks):
            result[t1.id].add(t1.id)
            for t2 in tracks[i + 1 :]:
                if is_overlap(t1.intervals, t2.intervals, threshold=threshold):
                    result[t1.id].add(t2.id)
                    result[t2.id].add(t1.id)
        self.contradictory_tracks = result
        return result

    def merge_confirmed_tracks(self) -> bool:
        logger.info("merging confirmed tracks...")
        sets = []
        for group in self.track_groups:
            group_ids = group["ids"]
            if len(group_ids) < self.max_det:
                break
            for id1 in group_ids:
                confirmed = (
                    set(range(len(self.tracks))) - self.contradictory_tracks[id1]
                )
                for id2 in group_ids:
                    if id1 == id2:
                        continue
                    confirmed &= self.contradictory_tracks[id2] - {id2}  # 需要剔除本身
                if len(confirmed):
                    confirmed.add(id1)
                    sets.append(confirmed)
        if len(sets):
            sets = merge_sets_with_common_elements(sets)
            for s in sets:
                try:
                    self.merge(s)
                except IntervalOverlapError as e:
                    logger.error(
                        f"failed to merge: {e}. This is not noraml, please check the code."
                    )
                    raise e
            self._clean(action="merge")
            return True
        else:
            logger.info("no confirmed tracks detected")
            return False

    def plot(self, title="", output_file=None):
        if self.plot_cnt == 0:
            try:
                shutil.rmtree(f"./exp_files/{VIDEO_NAME}/plots")
            except:
                pass
            Path(f"./exp_files/{VIDEO_NAME}/plots").mkdir(parents=True, exist_ok=True)
        logger.debug(f"plotting {self.plot_cnt} track plot...")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        for t in self.tracks:
            if t.assigned_class is None:
                color = "gray"
            else:
                color = random_color(t.assigned_class)
                color = (color[0] / 255, color[1] / 255, color[2] / 255)
            for interval in t.intervals:
                ax.plot([interval[0], interval[1]], [t.id] * 2, color=color)
            ax.text(t.intervals[0][0], t.id, t.id)
        ax.set_title(title)
        # if output_file:
        #   plt.savefig(output_file, bbox_inches='tight')
        Path(f"./exp_files/{VIDEO_NAME}/plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(
            f"./exp_files/{VIDEO_NAME}/plots/tracks_{self.plot_cnt}.png",
            bbox_inches="tight",
            dpi=300,
        )
        self.plot_cnt += 1

    def get_assigned_tracks(self):
        return sorted([t.merged for t in self.tracks if t.assigned_class is not None])

    def export_dataset(self, stride=MIN_STRIDE, split_train_val=0.1):
        self.new_imgs_cnt = 0
        self.old_imgs_cnt = 0
        import os
        from pathlib import Path

        train = {
            "metainfo": {"classes": [str(_) for _ in range(self.max_det)]},
            "data_list": [],
        }
        val = {
            "metainfo": {"classes": [str(_) for _ in range(self.max_det)]},
            "data_list": [],
        }
        if not self.dataset_exported:
            ids = self.track_groups[0]["ids"]
            assigned_classes = list(range(self.max_det))
        else:
            for g in self.track_groups:
                assigned_classes = [self.tracks[id].assigned_class for id in g["ids"]]
                if None in assigned_classes:
                    continue
                else:
                    ids = g["ids"]
                    logger.debug("assigned classes: " + str(assigned_classes))
                    break
        root = Path(f"./exp_files/{VIDEO_NAME}/cropped")
        frame_cnts = np.array([self.tracks[id].frame_cnt for id in ids])
        strides = [[] for _ in range(self.max_det)]
        data_lists = {
            "train": [[] for _ in range(self.max_det)],
            "val": [[] for _ in range(self.max_det)],
        }
        for i, id in enumerate(ids):
            extended_ids = self.tracks[id].merged
            if self.tracks[id].assigned_class is not None:
                label = self.tracks[id].assigned_class
            else:
                logger.debug("assigning classes...")
                label = i
                self.tracks[id].assigned_class = label
            for extended_id in extended_ids:
                _stride = max(
                    min(MAX_STRIDE, self.trained_times[extended_id] + 1), stride
                )
                fnames = sorted(
                    os.listdir(root / str(extended_id)),
                    key=lambda x: int(x.split(".")[0]),
                )
                if self.trained_times[extended_id] != 0:
                    random.shuffle(fnames)
                    self.old_imgs_cnt += len(fnames)
                else:
                    self.new_imgs_cnt += len(fnames)
                fnames = fnames[::_stride]
                if len(fnames) > 5000:
                    fnames = random.sample(fnames, 5000)
                split_point = int(len(fnames) * (1 - split_train_val))
                strides[i].append(_stride)
                self.trained_times[extended_id] += 1
                fnames = [f"{extended_id}/{img}" for img in fnames]
                data_lists["train"][label].extend(fnames[:split_point])
                data_lists["val"][label].extend(fnames[split_point:])
            logger.info(
                f"assigned class {label} to track {id} consists of {sorted(extended_ids)}"
            )

        train_lengths = np.array(
            [len(data_lists["train"][i]) for i in range(self.max_det)]
        )
        min_train_len = train_lengths.min()
        val_lengths = np.array([len(data_lists["val"][i]) for i in range(self.max_det)])
        min_val_len = val_lengths.min()
        logger.info(f"total frames for each class: {frame_cnts[assigned_classes]}")
        logger.info(f"train lengths: {train_lengths}")
        for i in range(self.max_det):
            # for img in random.sample(data_lists['train'][i], min_train_len): # 保证每个类别的样本数相同
            if len(data_lists["train"][i]) > 3600:
                data_lists["train"][i] = random.sample(data_lists["train"][i], 3600)
            if len(data_lists["val"][i]) > 400:
                data_lists["val"][i] = random.sample(
                    data_lists["val"][i], min(400, min_val_len)
                )
            for img in data_lists["train"][i]:
                train["data_list"].append({"img_path": img, "gt_label": i})
            for img in data_lists["val"][i]:
                val["data_list"].append({"img_path": img, "gt_label": i})
        self.dataset_exported = True
        if split_train_val:
            logger.info(
                f'total train samples: {len(train["data_list"])}, total val samples: {len(val["data_list"])}'
            )
            return train, val
        else:
            train["data_list"].extend(val["data_list"])
            logger.info(f'total samples: {len(train["data_list"])}')
            return train

    def _batch_inference(self, filepaths):
        from mmengine.registry import DefaultScope
        from mmengine.dataset import Compose, default_collate

        test_pipeline_cfg = self.cls_model.cfg.test_dataloader.dataset.pipeline
        if test_pipeline_cfg[0]["type"] != "LoadImageFromFile":
            test_pipeline_cfg.insert(0, dict(type="LoadImageFromFile"))
        with DefaultScope.overwrite_default_scope("mmcls"):
            test_pipeline = Compose(test_pipeline_cfg)
        scores_list = []
        for i in range(0, len(filepaths), BATCH_SIZE):
            batch = filepaths[i : min(i + BATCH_SIZE, len(filepaths))]
            data = [dict(img_path=img) for img in batch]
            data = [test_pipeline(d) for d in data]
            data = default_collate(data)
            with torch.no_grad():
                with cls_timer:
                    prediction = cls_model.val_step(data)
            scores = [p.pred_label.score.cpu().numpy() for p in prediction]
            scores = np.array(scores)
            scores_list.append(scores)
        return np.concatenate(scores_list)

    def get_classification_scores(self, stride: int) -> np.ndarray:
        scores = [[] for _ in self.tracks]
        min_score = float("inf")
        if self.cls_model_updated:
            for t in track_iter_progress(self.tracks):
                if t.frame_cnt <= 10:
                    scores[t.id] = np.zeros(self.max_det)
                    continue
                extended_ids = t.merged
                for id in extended_ids:
                    img_root = Path(f"./exp_files/{VIDEO_NAME}/cropped") / str(id)
                    fnames = os.listdir(img_root)
                    if len(fnames) < 32:
                        _stride = 1
                    else:
                        _stride = stride
                    paths = [str(img_root / fname) for fname in fnames[::_stride]]
                    result = self._batch_inference(paths)
                    scores[t.id].append(result)
                scores[t.id] = np.mean(np.concatenate(scores[t.id]), axis=0)
                t.pred = scores[t.id]
                if t.assigned_class is not None:
                    assert (
                        t.assigned_class == t.pred.argmax()
                    ), f"assigned class {t.assigned_class} not equal to pred {t.pred.argmax()}"
                    logger.debug(
                        f"{t.id} (original {t.original_id}) class {t.assigned_class} score is {t.pred} ({t.pred[t.assigned_class]})"
                    )
                    min_score = min(min_score, t.pred[t.assigned_class])

            logger.info("all cropped images forwarded to DL model.")
            self.cls_model_updated = False
        else:
            scores = np.empty((len(self.tracks), self.max_det))
            for t in self.tracks:
                if t.frame_cnt <= 10:
                    scores[t.id] = np.zeros(self.max_det)
                else:
                    scores[t.id] = t.pred
        scores = np.array(scores)
        assert np.isnan(scores).any() == False, f"nan in scores: {scores}"
        return scores, min_score

    def process_dumped(self):
        logger.info("processing dumped tracks...")
        scores = [[] for _ in self.dumped]
        max_id = len(self.tracks) - 1
        self.dumped.sort(key=lambda x: x.intervals[0][0])
        for i, t in enumerate(self.dumped):
            extended_ids = t.merged
            for id in extended_ids:
                img_root = Path(f"./exp_files/{VIDEO_NAME}/cropped") / str(id)
                fnames = os.listdir(img_root)
                paths = [str(img_root / fname) for fname in fnames]
                result = self._batch_inference(paths)
                scores[i].append(result)
            scores[i] = np.mean(np.concatenate(scores[i]), axis=0)
            t.pred = scores[i]
            # re-id dumped tracks
            t.id = max_id + i + 1
            self.tracks.append(t)
        self.tracks.sort(key=lambda x: x.intervals[0][0])
        for i, t in enumerate(self.tracks):
            t.id = i
        logger.debug(
            f"tracks info after dumped {len(self.tracks)}, {self.tracks[-1].id}"
        )
        self.dumped = []
        # t.assigned_class = t.pred.argmax()

    def force_merge(self):
        logger.info("force merging...")
        self.update_contradictory_tracks()
        self.update_tracks_groups()
        self.merge_confirmed_tracks()
        for g in self.track_groups:
            ids = g["ids"]
            assigned_classes = [self.tracks[id].assigned_class for id in ids]
            if None in assigned_classes:
                continue
            else:
                break
        assert None not in assigned_classes, f"assigned classes: {assigned_classes}"
        target = ids

        for g in self.track_groups:
            ids = g["ids"]
            assigned_classes = [self.tracks[id].assigned_class for id in ids]
            if None not in assigned_classes:
                continue
            dist_mat = self._compute_reid_costmat(ids, target)
            matches = linear_assignment(dist_mat)
            for i, j in matches:
                if self.tracks[g["ids"][i]].assigned_class is None:
                    self.tracks[g["ids"][i]].assigned_class = self.tracks[
                        target[j]
                    ].assigned_class
        # TODO: 先进行合并
        to_be_merged = [set() for _ in range(self.max_det)]
        for t in self.tracks:
            if t.assigned_class is None:
                continue
            to_be_merged[t.assigned_class].add(t.id)
        cleaned_sets = self.remove_contradict(to_be_merged)
        for i in range(self.max_det):
            logger.debug(
                f"Some tracks are contradictory: {to_be_merged[i] - cleaned_sets[i]}"
            )
            for id in to_be_merged[i] - cleaned_sets[i]:
                self.tracks[id].assigned_class = None
        for s in cleaned_sets:
            self.merge(s)
        self._clean("force merged")
        self.update_contradictory_tracks()
        updated = self.update_tracks_groups(discard=True)
        while not updated:
            self.update_contradictory_tracks()
            updated = self.update_tracks_groups(discard=True)
        merged = track_merger.merge_confirmed_tracks()
        while merged:
            self.update_contradictory_tracks()
            self.update_tracks_groups()
            merged = track_merger.merge_confirmed_tracks()

        logger.info(
            "force merged more valuable trackes. now merging short and questionable tracks..."
        )
        self.plot("final")

        self.process_dumped()
        self.plot("dump")
        self.update_contradictory_tracks(SOFT_BORDER)
        updated = self.update_tracks_groups(discard=True)
        while not updated:
            self.update_contradictory_tracks(SOFT_BORDER)
            updated = self.update_tracks_groups(discard=True)
        logger.debug(f"track groups are {self.track_groups}")
        self.merge_confirmed_tracks()
        self.update_contradictory_tracks(SOFT_BORDER)
        updated = self.update_tracks_groups(discard=True)
        assert updated, "track groups not updated"
        logger.debug(f"tracks are {[t.id for t in self.tracks]}")
        for g in self.track_groups:
            ids = g["ids"]
            assigned_classes = [self.tracks[id].assigned_class for id in ids]
            if None in assigned_classes:
                continue
            else:
                break
        assert None not in assigned_classes, f"assigned classes: {assigned_classes}"
        target = ids

        for g in self.track_groups:
            ids = g["ids"]
            logger.debug(f"ids: {ids}, len tracks: {len(self.tracks)}")
            assigned_classes = [self.tracks[id].assigned_class for id in ids]
            if None not in assigned_classes:
                continue
            dist_mat = self._compute_reid_costmat(ids, target)
            matches = linear_assignment(dist_mat)
            for i, j in matches:
                if dist_mat[i, j] > 0.3:  # DEBUG 消除质量太差的匹配
                    logging.debug(f"cost too high: {dist_mat[i, j]}")
                    continue
                if self.tracks[g["ids"][i]].assigned_class is None:
                    self.tracks[g["ids"][i]].assigned_class = self.tracks[
                        target[j]
                    ].assigned_class

        for t in self.dumped:
            if t.frame_cnt > 150:
                t.assigned_class = np.argmax(
                    t.pred
                ).item()  # np.int64type caused error in sqlite
        self.finished = True

    def handle_contradiction(self, id):
        self.tracks[id].triggered_contradictions += 1
        logger.info(
            f"handling contradiction caused by {self.tracks[id]}... It has triggered {self.tracks[id].triggered_contradictions} times."
        )
        if self.tracks[id].triggered_contradictions >= 3:
            logger.info(
                f"{self.tracks[id]} has triggered {self.tracks[id].triggered_contradictions} times, remove it."
            )
            self._trash.append(id)

    def classify_uncertain_tracks(
        self, prob_threshold=0.7, stride=INFERENCE_STRIDE
    ) -> Union[bool, int]:
        logger.info(
            f"classifying uncertain tracks with prob threshold {prob_threshold}..."
        )
        scores, min_score = self.get_classification_scores(stride)
        classified_frames = 0
        total_frames = 0
        for t in self.tracks:
            if t.assigned_class is not None:
                classified_frames += t.frame_cnt
            total_frames += t.frame_cnt
        logger.info(
            f"{classified_frames} frames are classified, {total_frames} frames in total ({classified_frames/total_frames:.2%})"
        )
        if classified_frames / total_frames > ACCUMULATED_FRAME_RATIO:
            logger.info(
                r"95% of frames are classified, skip classification and start merging..."
            )
            self.force_merge()
            return True

        # if min_score < prob_threshold:
        #   prob_threshold = min_score # 防止出现片段被分类而并没有和主干合并的情况
        #   logger.info(f'set prob threshold to min score: {min_score}')
        # 此处好像有bug：有些分数高于prob threshold但是没有被加入confirmed sets
        # 另外会产生离群的小片段
        max_scores = scores.max(axis=1)
        order = max_scores.argsort()[::-1]
        max_cls = scores.argmax(axis=1)
        logger.debug(f"classification scores:\n{scores[order][:min(10, len(scores))]}")
        logger.debug(f"max cls {max_cls[order][:min(10, len(scores))]}")
        confirmed_sets = [set() for _ in range(self.max_det)]
        stop_flag = False
        for t in self.tracks:
            if t.assigned_class is not None:
                confirmed_sets[t.assigned_class].add(t.id)
        for i in order:
            if max_scores[i] < prob_threshold:
                break
            cls = max_cls[i]
            for t in confirmed_sets[cls]:
                if i in self.contradictory_tracks[t] and i != t:
                    logger.info(
                        f"contradiction ({self.tracks[i]} and {self.tracks[t]}) detected before prob threshold is reached {prob_threshold}. min prob is {max_scores[i]}"
                    )
                    self.handle_contradiction(i)
                    stop_flag = True
                    break

            if not stop_flag:
                confirmed_sets[cls].add(i)
            else:
                break

        if not stop_flag:
            logger.info(
                f"Reaching prob threshold {prob_threshold} without contradictions"
            )

        newly_classified_frames = 0
        logger.debug(f"confirmed sets: {confirmed_sets}")
        for cls, s in enumerate(confirmed_sets):
            if len(s) > 1:
                for id in s:
                    if self.tracks[id].assigned_class is None:
                        newly_classified_frames += self.tracks[id].frame_cnt
                    self.tracks[id].assigned_class = cls
                self.merge(s)
        if newly_classified_frames > 0:
            logger.info(f"{newly_classified_frames} frames are newly classified")
        self._clean(action="classify")
        return newly_classified_frames

    def differentiate(self):
        global APPEARANCE_THRESHOLD
        logger.warning(
            "Classification phase failed, trying to differentiate tracks, "
            f"which may be unreliable"
        )
        to_be_merged = []
        to_be_merged_dist = []
        groups_to_be_differentiated = []
        for i, g1 in enumerate(self.track_groups):
            for j, g2 in enumerate(self.track_groups[i + 1 :]):
                g1_ids = set(g1["ids"])
                g2_ids = set(g2["ids"])
                intersection = g1_ids & g2_ids
                if len(intersection) > 0:
                    groups_to_be_differentiated.append(
                        (list(g1_ids - intersection), list(g2_ids - intersection))
                    )
        min_dist = float("inf")
        best_match = None
        total_dists = []
        for g1, g2 in groups_to_be_differentiated:
            assigned_classes = [self.tracks[id].assigned_class for id in g1]
            if None in assigned_classes:
                assigned_classes = [self.tracks[id].assigned_class for id in g2]
                if None in assigned_classes:
                    continue
            dist_mat = self._compute_reid_costmat(g1, g2)
            matches = linear_assignment(dist_mat)
            total_dist = 0
            for i, j in matches:
                total_dist += dist_mat[i, j]
                total_dists.append(total_dist)
            if total_dist / len(matches) < min_dist:
                min_dist = total_dist / len(matches)
                best_match = [{g1[i], g2[j]} for i, j in matches]
            if total_dist / len(matches) < APPEARANCE_THRESHOLD:
                to_be_merged.append([{g1[i], g2[j]} for i, j in matches])
                to_be_merged_dist.append(total_dist)
                logger.info(f"good match found: {[{g1[i], g2[j]} for i, j in matches]}")
        if len(to_be_merged) == 0 and best_match is not None:
            APPEARANCE_THRESHOLD *= 1.15
            logger.warning(
                f"no good match found, merging lowest cosine dist {min_dist}. We are now more tolerant to unclassified tracks by increasing APPEARANCE_THRESHOLD 10% to {APPEARANCE_THRESHOLD}"
            )
            to_be_merged.append(best_match)
            to_be_merged_dist.append(min_dist)
        # sets = merge_sets_with_common_elements(to_be_merged.copy())
        # logger.debug(f'to be merged sets: {to_be_merged} ==> {sets}')
        # clean_sets = self.remove_contradict(sets.copy())
        # logger.debug(f'cleaned sets: {sets} ==> {clean_sets}')
        # logger.debug(f'total matching cosine dists: {total_dists}')
        to_be_merged = [s for _, s in sorted(zip(to_be_merged_dist, to_be_merged))]
        to_be_merged_dist = sorted(to_be_merged_dist)
        accumulated_frames = 0
        final = []
        final_dist = []
        for i, group in enumerate(to_be_merged):
            for s in group:
                for id in s:
                    if self.tracks[id].assigned_class is None:
                        accumulated_frames += self.tracks[id].frame_cnt
            final.extend(group)
            final_dist.append(to_be_merged_dist[i])
            # if accumulated_frames > MIN_MERGE_FRAMES:
            #   break
        # logger.debug(f'currently only merging best match: {min_dist} {best_match}')
        # sets = merge_sets_with_common_elements(best_match)
        sets = merge_sets_with_common_elements(final)
        clean_sets = self.remove_contradict(sets)
        logger.debug(
            f"merging {clean_sets} with dist {final_dist}. total frames: {accumulated_frames}"
        )
        for s in clean_sets:
            self.merge(s)
        self._clean(action="differentiate")

    def remove_contradict(self, l_sets: list):
        l_sets = [set(s) for s in l_sets]
        for s in l_sets:
            contradictions = set()
            for id in s:
                intersection = s & self.contradictory_tracks[id]
                if len(intersection) > 1:
                    contradictions |= intersection
            s -= contradictions
        return l_sets

    def _compute_reid_costmat(self, g1, g2):
        from constants import DISTANCE_TYPE
        from scipy.spatial import distance

        mat = np.zeros((len(g1), len(g2)))
        for i in range(len(g1)):
            for j in range(len(g2)):
                if DISTANCE_TYPE == "cosine":
                    mat[i, j] = distance.cosine(
                        self.tracks[g1[i]].pred, self.tracks[g2[j]].pred
                    )
                elif DISTANCE_TYPE == "jensen-shannon" or DISTANCE_TYPE == "js":
                    mat[i, j] = distance.jensenshannon(
                        self.tracks[g1[i]].pred, self.tracks[g2[j]].pred
                    )
                else:
                    logger.warning(
                        f"unknown distance type {DISTANCE_TYPE}, using cosine distance instead"
                    )
                    mat[i, j] = distance.cosine(
                        self.tracks[g1[i]].pred, self.tracks[g2[j]].pred
                    )
        logger.debug(f"cosine dist between {g1} and {g2}: {mat}")
        return mat

    def output_id_mapper(self, min_frame_cnt=10):
        id_mapper = {}
        for t in self.dumped:
            id_mapper[t.id] = t.assigned_class if t.assigned_class is not None else t.id
        for t in self.tracks:
            label = t.assigned_class + 1 if t.assigned_class is not None else t.id
            if t.frame_cnt < min_frame_cnt:
                label = 0
            for id in t.merged:
                id_mapper[id] = label
        with open(f"./exp_files/{VIDEO_NAME}/idmapper.pkl", "wb") as f:
            pickle.dump(id_mapper, f)
        return id_mapper

    def save_state(self):
        import pickle

        temp_model = self.cls_model
        self.cls_model = None
        p = Path(f"./exp_files/{VIDEO_NAME}/backup")
        if not getattr(self, "saved_state", False) and p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / f"backup_{self.backup_cnt:0>6}.pkl", "wb") as f:
            pickle.dump(self, f)
        self.backup_cnt += 1
        self.cls_model = temp_model

    @staticmethod
    def restore(backup_id) -> "TrackMerger":
        import pickle

        with open(
            f"./exp_files/{VIDEO_NAME}/backup/backup_{backup_id:0>6}.pkl", "rb"
        ) as f:
            backup = pickle.load(f)
        return backup

    def is_finished(self):
        if getattr(self, "finished", False):
            return True
        cnt = 0
        for t in self.tracks:
            if t.frame_cnt > 10:
                cnt += 1
                if cnt > self.max_det:
                    self.finished = False
                    return False
        assigned_classes = set([t.assigned_class for t in self.tracks])
        if None in assigned_classes:
            self.finished = False
            return False
        self.finished = True
        if self.finished:
            self.force_merge()
        return True


from itertools import combinations


def merge_sets_with_common_elements(sets: List[set]):
    # https://stackoverflow.com/questions/27180877/how-can-i-merge-lists-that-have-common-elements-within-a-list-of-lists
    sets = [set(s) for s in sets]
    fixed_point = False
    while not fixed_point:
        fixed_point = True
        for x, y in combinations(sets, 2):
            if x & y:
                x.update(y)
                sets.remove(y)
                fixed_point = False
                break
    return sets


if __name__ == "__main__":
    Session = sessionmaker(bind=engine)
    if resume > 0:
        logger.info(f"restore {resume}")
        track_merger = TrackMerger.restore(resume)
    else:
        session = Session()
        tracks_info = (
            session.query(TrackletStat)
            .filter(TrackletStat.video_id == VIDEO_ID)
            .order_by(TrackletStat.track_id)
            .all()
        )

        l = []
        for t in tracks_info:
            t_intervals = ast.literal_eval(t.intervals)
            intervals = []
            for i in range(0, len(t_intervals), 2):
                intervals.append((t_intervals[i], t_intervals[i + 1]))
            l.append(
                Track(
                    t.start_frame,
                    t.end_frame,
                    intervals,
                    t.track_id,
                    t.conf,
                    t.distance,
                )
            )
        print(f"total tracks: {len(tracks_info)}")
        track_merger = TrackMerger(l, max_det=MAX_DET)
        track_merger.plot("start")

    fixed_tracks = False
    num_tracks = float("inf")
    num_last_train = 0

    classified = False
    while not fixed_tracks:
        merged = True
        while 1:
            track_merger.update_contradictory_tracks()
            not_discarded = track_merger.update_tracks_groups(
                discard=True
            )  # TODO: maybe remove discard True?
            if not not_discarded:
                continue
            track_merger.merge_confirmed_tracks()
            break
        track_merger.update_contradictory_tracks()
        track_merger.update_tracks_groups()
        title = "classification" if classified else "reid"
        track_merger.plot(title)
        if track_merger.is_finished():
            logger.info(
                f"track merging finished, total tracks: {len(track_merger.tracks)}"
            )
            break
        else:
            if num_tracks == len(track_merger.tracks):
                fixed_tracks = False
                logger.info("track num fixed, exit running")
                # break
            else:
                num_tracks = len(track_merger.tracks)

        train, val = track_merger.export_dataset(stride=MIN_STRIDE)
        if len(train["data_list"]) % BATCH_SIZE == 1:
            logger.warning("last train sample is discarded")
            train["data_list"] = train["data_list"][:-1]
        if len(val["data_list"]) % BATCH_SIZE == 1:
            logger.warning("last val sample is discarded")
            val["data_list"] = val["data_list"][:-1]
        with open(f"./exp_files/{VIDEO_NAME}/cropped/train.json", "w") as f:
            json.dump(train, f)
        with open(f"./exp_files/{VIDEO_NAME}/cropped/val.json", "w") as f:
            json.dump(val, f)
        metrics, best_cls_model_path = track_merger.update_cls_model(
            len(train["data_list"])
        )
        logger.info(f'accuracy/top1: {metrics["accuracy/top1"]}')
        if metrics["accuracy/top1"] < 0.4:
            raise UnreliableClassificationModelError(metrics["accuracy/top1"])

        cfg = track_merger.build_cfg(CLS_CONFIG_FILE)
        cfg.load_from = best_cls_model_path
        cls_model = init_model(cfg, best_cls_model_path, "cuda:0")
        track_merger.cls_model = cls_model

        backup_id = track_merger.backup_cnt
        track_merger.save_state()
        prob_threshold = 0.9
        tried_threshod = set()
        while (
            prob_threshold >= 0.85
            and prob_threshold < 1
            and prob_threshold not in tried_threshod
        ):
            tried_threshod.add(prob_threshold)
            classified = track_merger.classify_uncertain_tracks(
                prob_threshold, MIN_STRIDE
            )
            if track_merger.finished:
                break
            tmp = [[] for _ in range(MAX_DET)]
            for t in track_merger.tracks:
                if t.assigned_class:
                    tmp[t.assigned_class].append(t)
                    assert (
                        len(tmp[t.assigned_class]) == 1
                    ), f"assigned class error for class {t.assigned_class}, track {tmp[t.assigned_class]}"
            if classified:
                try:
                    track_merger.update_contradictory_tracks()
                    if not track_merger.update_tracks_groups(discard=True):
                        track_merger.update_contradictory_tracks()
                        track_merger.update_tracks_groups()
                    track_merger.merge_confirmed_tracks()
                    track_merger.update_contradictory_tracks()
                    track_merger.update_tracks_groups()
                    if classified > MIN_MERGE_FRAMES:
                        break
                    else:
                        prob_threshold -= 0.05
                except (InvalidCliqueError, ConflictAssignedClassError) as e:
                    logger.error(e)
                    prob_threshold += 0.025
                    logger.warning(
                        f"Rolling back to original state and increase prob threshold to {prob_threshold}..."
                    )
                    track_merger = TrackMerger.restore(backup_id)
                    track_merger.cls_model = cls_model
            else:
                prob_threshold -= 0.05

        if classified:
            if track_merger.finished:
                break
            if classified > MIN_MERGE_FRAMES:
                continue
            else:
                track_merger.update_contradictory_tracks()
                track_merger.update_tracks_groups()
                track_merger.differentiate()
                track_merger.plot("reid-before-merge")
        else:
            track_merger.update_contradictory_tracks()
            track_merger.update_tracks_groups()
            track_merger.differentiate()
            track_merger.plot("reid-before-merge")

    logger.info(f"total training time: {train_timer.t:.2f}s")
    logger.info(f"total classification time: {cls_timer.t:.2f}s")
    logger.info(f"total grouping time: {group_track_timer.t:.2f}s")

    id_mapper = track_merger.output_id_mapper(0)
    for i in range(len(track_merger._original_tracks)):
        data = session.query(Datum).filter(
            Datum.video_id == VIDEO_ID, Datum.raw_track_id == i
        )
        for d in data:
            id = id_mapper.get(i, None)
            if id is not None and id <= MAX_DET:
                d.track_id = id
        session.commit()
    # generate_video(track_merger)
