from app.algorithm.classifier import MMPretrainClassifier
from constants import SOFT_BORDER
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
import networkx as nx
import random
import json
from app.algorithm.common import match
import scipy.spatial.distance as scipy_distance
from app.algorithm.models import TrackletStat
from mmengine.utils import track_iter_progress
import shutil
import mmcv


def merge_sets_with_common_elements(sets: list[set]):
    # https://stackoverflow.com/questions/27180877/how-can-i-merge-lists-that-have-common-elements-within-a-list-of-lists
    from collections import combinations

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


@dataclass
class Interval:
    id: int  # original id of the tracklet from the database
    start: int
    end: int  # inclusive


class Tracklet:
    def __init__(self, intervals: list[Interval], assigned_id=None) -> None:
        self.intervals: list[Interval] = intervals
        self.pred_score: np.ndarray = None
        self.assigned_id = assigned_id  # track id assigned by merger, start from 0
        self.triggered_conflicts = 0

    def remove_by_id(self, id: int):
        self.intervals = [x for x in self.intervals if x.id != id]

    def get_ids(self) -> set[int]:
        return set([x.id for x in self.intervals])

    def get_frames(self) -> list[int]:
        return list(range(self.intervals[0].start, self.intervals[-1].end + 1))

    def __add__(self, other):
        # TODO: 有可能合并一些重叠的interval，需要改进
        intervals = sorted(self.intervals + other.intervals, key=lambda x: x.start)
        assert (
            self.assigned_id is None
            or other.assigned_id is None
            or self.assigned_id == other.assigned_id
        ), f"assigned_id not equal {self.assigned_id} != {other.assigned_id}"
        assigned_id = (
            self.assigned_id if self.assigned_id is not None else other.assigned_id
        )
        return Tracklet(intervals, assigned_id)

    def __iadd__(self, other):
        assert (
            self.assigned_id is None
            or other.assigned_id is None
            or self.assigned_id == other.assigned_id
        ), f"assigned_id not equal {self.assigned_id} != {other.assigned_id}"
        self.intervals = sorted(self.intervals + other.intervals, key=lambda x: x.start)
        self.assigned_id = (
            self.assigned_id if self.assigned_id is not None else other.assigned_id
        )
        return self

    def __len__(self):
        return sum([x.end - x.start + 1 for x in self.intervals])


def detect_overlap(
    t1: Tracklet,
    t2: Tracklet,
    threshold: int = SOFT_BORDER,
    boolean: bool = False,
):
    i, j = 0, 0
    result = []
    while i < len(t1.intervals) and j < len(t2.intervals):
        if t1.intervals[i].end < t2.intervals[j].start:
            i += 1
        elif t1.intervals[j].end < t2.intervals[i].start:
            j += 1
        else:
            overlap_length = (
                min(t1.intervals[i].end, t2.intervals[j].end)
                - max(t1.intervals[i].start, t2.intervals[j].start)
                + 1
            )
            if overlap_length >= threshold:
                if boolean:
                    return True
                else:
                    if t1.intervals[i].id < t2.intervals[j].id:
                        result.append((t1.intervals[i].id, t2.intervals[j].id))
                    else:
                        result.append((t2.intervals[j].id, t1.intervals[i].id))
            else:
                if t1.intervals[i].end < t2.intervals[j].end:
                    i += 1
                else:
                    j += 1
    if boolean:
        return False
    else:
        result = list(set(result))
        return result


class Merger:
    def __init__(
        self,
        tracks: list[Tracklet],
        max_det: int,
        cls_model: MMPretrainClassifier,
        img_root: Path,
        exp_root: Path,
        video_path: str,
        logger,
        session,
        video_id,
        confidence_threshold: float = 0.95,
        soft_border: int = SOFT_BORDER,
        train_ratio=0.9,
        max_frames_per_class=6000,
        batch_size=256,
        min_merge_frames=150,
        apperance_threshold=0.05,
        min_confidence=0.6,
    ) -> None:
        self.tracklets: list[Tracklet] = sorted(tracks, lambda x: x.intervals[0].start)
        # tracklets that are assigned an id, sorted by Tracklet.assigned_id
        self.assigned_tracklets: list[Tracklet] = []
        # tracklets that are dumped
        self.dumped: list[Tracklet] = []
        self.max_det = max_det
        self.model = cls_model
        self.session = session
        self.video_id = video_id
        self.img_root = img_root
        self.exp_root = exp_root
        self.video_path = video_path
        self.conf_threshold = confidence_threshold
        self.soft_border = soft_border  # dynamic
        self._soft_border = soft_border
        self.logger = logger
        self.train_ratio = train_ratio
        self.max_frames_per_class = max_frames_per_class
        self.batch_size = batch_size
        self.min_merge_frames = min_merge_frames
        self.apperance_threshold = apperance_threshold  # dynamic
        self._appearance_threshold = apperance_threshold
        self.min_confidence = min_confidence

        self.accumulated_new_frames = 0
        self.trained_times = {}  # tracklet.id -> number of times trained

        self.logger.info(f"total # of tracklets: {len(self.tracklets)}")
        self.filter_tracks()

    def merge(self) -> None:
        self.init_tracks_kernel()
        self.update_cls_model(patience=20)
        while (
            len(self.tracklets) > self.max_det
            or self.get_assigned_length() < self.get_total_length() * self.train_ratio
        ):
            self.merge_by_appearance()
            self.merge_confirmed_tracklets()
            self.update_cls_model(patience=self.calculate_patience())

        if len(self.tracklets) > self.max_det:
            self.process_dumped()

    def process_dumped(self):
        self.tracklets.extend(self.dumped)
        self.tracklets.sort(key=lambda x: x.intervals[0].start)
        self.merge_confirmed_tracklets()
        self.merge_by_appearance()
        self.merge_confirmed_tracklets()

    def update_cls_model(self, patience: int):
        train, val = self.generate_dataset()
        with open(self.img_root / "train.json", "w") as f:
            json.dump(train, f)
        with open(self.img_root / "val.json", "w") as f:
            json.dump(val, f)
        self.model.train(patience=patience)

    def filter_tracks(self) -> None:
        if len(mean_areas) <= self.max_det * 2:
            self.logger.info("too few tracks, skip track filtering")
            return
        # filter by confidence
        tracklet_stats = np.array(
            self.session.query(
                TrackletStat.mask_area,
                TrackletStat.lifespan,
                TrackletStat.distance,
                TrackletStat.conf,
            )
            .filter(TrackletStat.video_id == self.video_id)
            .order_by(TrackletStat.track_id)
            .all()
        )
        mean_areas, mean_weights, mean_distance, mean_conf = (
            tracklet_stats[:, 0].ravel(),
            tracklet_stats[:, 1].ravel(),
            tracklet_stats[:, 2].ravel(),
            tracklet_stats[:, 3].ravel(),
        )
        mean_area = np.average(mean_areas, weights=mean_weights)
        std_area = np.sqrt(np.cov(mean_areas, aweights=mean_weights))
        speeds = mean_distance / mean_weights
        mean_speed = np.average(speeds, weights=mean_weights)
        std_speed = np.sqrt(np.cov(speeds, aweights=mean_weights))

        area_threshold = mean_area + 3 * std_area
        area_threshold_low = mean_area - 3 * std_area
        too_large_area = list(np.arange(len(self.tracks))[mean_areas > area_threshold])
        too_small_area = list(
            np.arange(len(self.tracks))[mean_areas < area_threshold_low]
        )
        too_short = list(np.arange(len(self.tracks))[mean_weights <= self.soft_border])
        speed_threshold = mean_speed - 3 * std_speed
        too_slow = list(np.arange(len(self.tracks))[speeds < speed_threshold])
        not_confident = list(
            np.arange(len(self.tracks))[mean_conf < self.conf_threshold]
        )

        self.logger.info(f"mean area: {mean_area}, std area: {std_area}")
        self.logger.info(f"mean speed: {mean_speed}, std speed: {std_speed}")
        self.logger.info(f"too large mean area: {len(too_large_area)}")
        self.logger.info(f"too small mean area: {len(too_small_area)}")
        self.logger.info(f"too short: {len(too_short)}")
        self.logger.info(f"too slow: {len(too_slow)}")
        self.logger.info(f"not confident enough: {len(not_confident)}")
        to_be_removed = set(
            too_short + not_confident + too_large_area + too_slow + too_small_area
        )
        self.logger.info(f"total # to be removed: {len(to_be_removed)}")
        self.remove_tracklets(to_be_removed, remove=True)

    def export_difficult_frames(self, ids: list[int]):
        save_root_path = self.exp_root / "difficult"
        self.logger.info(
            f"exporting difficult frames... You may need to annotate them for better results. Images will be saved to {str(save_root_path)}"
        )

        if save_root_path.exists():
            shutil.rmtree(save_root_path)
        save_root_path.mkdir(parents=True, exist_ok=True)
        frames = []
        for id in ids:
            frames += self.tracklets[id].get_frames()
        frames = sorted(list(set(frames)))
        video = mmcv.VideoReader(self.video_path)
        step = video.fps * 4
        for i in range(0, len(frames), step):
            frame = video[i]
            mmcv.imwrite(frame, str(save_root_path / f"{i}.jpg"))

    def get_assigned_length(self) -> int:
        return sum([len(x) for x in self.assigned_tracklets])

    def get_total_length(self) -> int:
        return sum([len(x) for x in self.tracklets])

    def calculate_patience(self) -> int:
        patience = 3 * round(
            1 + self.accumulated_new_frames * 8 / self.get_assigned_length()
        )
        return patience

    def init_tracks_kernel(self) -> None:
        self.merge_confirmed_tracklets()
        cliques, to_be_removed = self.detect_cliques(
            self.get_conflict_list(self.soft_border)
        )
        assert (
            len(to_be_removed) == 0
        ), f"to_be_removed should be empty, but get {to_be_removed}"
        c = cliques[0]
        assert len(c) == self.max_det, f"len(c) should be {self.max_det}, but get {c}"
        for assigned_id, index in enumerate(c):
            self.tracklets[index].assigned_id = assigned_id
            self.assigned_tracklets.append(self.tracklets[index])
        self.assigned_tracklets.sort(key=lambda x: x.assigned_id)

    def generate_dataset(self):
        train = {
            "metainfo": {"classes": [str(_) for _ in range(self.max_det)]},
            "data_list": [],
        }
        val = {
            "metainfo": {"classes": [str(_) for _ in range(self.max_det)]},
            "data_list": [],
        }

        for t in self.assigned_tracklets:
            ids = t.get_ids()
            train_weights = []
            val_weights = []
            train_datalist = []
            val_datalist = []
            for id in ids:
                fpaths = [
                    str(self.img_root / str(id) / x)
                    for x in os.listdir(self.img_root / f"{id}")
                ]
                split_point = int(len(fpaths) * self.train_ratio)
                self.trained_times[id] = self.trained_times.get(id, 0) + 1
                datalist = [
                    {"img_path": fpath, "gt_label": t.assigned_id}
                    for fpath in fpaths[:split_point]
                ]
                weights += [1 / self.trained_times[id]] * len(datalist)
                if self.trained_times[id] > 1:
                    datalist = random.shuffle(datalist)
                train_datalist.extend(datalist[:split_point])
                val_datalist.extend(datalist[split_point:])
                train_weights.extend(weights[:split_point])
                val_weights.extend(weights[split_point:])
            if len(datalist) > self.max_frames_per_class:
                train_datalist = random.choices(
                    train_datalist,
                    weights=train_weights,
                    k=self.max_frames_per_class * self.train_ratio,
                )
                val_datalist = random.choices(
                    val_datalist,
                    weights=val_weights,
                    k=self.max_frames_per_class * (1 - self.train_ratio),
                )
            elif len(datalist) < self.batch_size:
                train_datalist = random.choices(
                    train_datalist, weights=train_weights, k=self.batch_size
                )
                val_datalist = random.choices(
                    val_datalist, weights=val_weights, k=self.batch_size
                )
            train["data_list"].extend(train_datalist)
            val["data_list"].extend(val_datalist)
        self.logger.info(
            f'total train data: {len(train["data_list"])}, total val data: {len(val["data_list"])}'
        )

        self.accumulated_new_frames = 0
        return train, val

    def get_conflict_list(self, soft_border: int) -> list[set[int]]:
        """get conflict list

        Args:
            soft_border (int): soft border for spatio-temporal overlap

        Returns:
            list[set[int]]: conflict_list[i] is the set of tracklets that conflict with tracklet i
        """
        result = [set() for _ in range(len(self.tracklets))]
        for i, t1 in enumerate(self.tracklets):
            for j, t2 in enumerate(self.tracklets[i + 1 :]):
                if i == j:
                    continue
                if detect_overlap(t1, t2, threshold=soft_border, boolean=True):
                    result[i].add(i + j + 1)
                    result[i + j + 1].add(i)
        return result

    def detect_cliques(
        self, conflict_list: list[set[int]]
    ) -> tuple[list[dict], list[int]]:
        """detect cliques in conflict_list

        Args:
            conflict_list (list[set[int]]): conflict_list[i] is the set of tracklets that conflict with tracklet i

        Returns:
            tuple[list[dict], list[int]]:
                cliques (ordered by minimum length in a clique), {"ids": list[int], "lifespan": int}
                to_be_removed (detected cliques that are longer than max_det)
        """
        G = nx.Graph()
        for i, s in enumerate(conflict_list):
            G.add_node(i)
            for j in s:
                G.add_edge(i, j)
        cliques = nx.find_cliques(G)
        to_be_removed = []
        result = []
        for c in cliques:
            if len(c) > self.max_det:
                self.logger.error(
                    f"detected clique has length {len(c)} > {self.max_det} (max det)"
                )
                c = sorted(c, key=lambda x: len(self.tracklets[x]), reverse=True)
                to_be_removed += c[self.max_det :]
                c = c[: self.max_det]
            lifespan = min(c, key=lambda x: len(self.tracklets[x]))
            result.append({"ids": c, "lifespan": lifespan})
        return result, to_be_removed

    def remove_tracklets(
        self, merged: list[int], to_be_removed: list[int], remove=False
    ) -> None:
        """remove tracklets in to_be_removed from self.tracklets

        Args:
            merged (list[int]): list of ids (in self.tracklets) that are merged and should be removed from self.tracklets
            to_be_removed (list[int]): list of ids (in self.tracklets) that are discarded and should be placed in self.dumped
        """
        self.dumped += [self.tracklets[i] for i in to_be_removed]
        need_to_be_removed = merged + to_be_removed
        assert len(need_to_be_removed) == len(set(need_to_be_removed))
        self.tracklets = [
            t for i, t in enumerate(self.tracklets) if i not in need_to_be_removed
        ]

    def merge_confirmed_tracklets(self) -> bool:
        """merge tracklets that are confirmed by spatio-temporal overlap

        Returns:
            bool: True if tracklets are merged
        """
        while self.soft_border < 60 * 10:
            conflict_list = self.get_conflict_list(self.soft_border)
            cliques, to_be_removed = self.detect_cliques(conflict_list)
            sets = []
            for c in cliques:
                ids = c["ids"]
                for id1 in ids:
                    confirmed = set(range(len(self.tracklets))) - conflict_list[id1]
                    for id2 in ids:
                        if id2 == id1:
                            continue
                        confirmed = confirmed.intersection(conflict_list[id2])
                    if (
                        len(confirmed) == 2
                    ):  # the tracklet itself and the other tracklet
                        assert id1 in confirmed, f"{id1} not in {confirmed}"
                        sets.append(confirmed)
            if len(sets):
                sets = merge_sets_with_common_elements(sets)
                if self.exist_assigned_id_conflict(sets):
                    self.soft_border = round(self.soft_border * 1.5)
                    self.logger.error(
                        f"assigned_id conflict detected! lossen soft border to {self.soft_border}"
                    )
                    continue
                self.soft_border = self._soft_border
                self._merge(sets, to_be_removed)
                return True
            else:
                self.logger.info("No tracklets to be merged by spatio-temporal overlap")
                return False
        raise Exception(f"soft border too large: {self.soft_border}. Catasrophic error")

    def exist_assigned_id_conflict(self, to_be_merged: list[set[int]]) -> bool:
        """check if there is assigned_id conflict in to_be_merged. This only happens when merging by spatio-temporal overlap

        Args:
            to_be_merged (list[set[int]]): list of ids to be merged together (in self.tracklets)

        Returns:
            bool: True if there is assigned_id conflict
        """
        for s in to_be_merged:
            assigned_ids = set([self.tracklets[x].assigned_id for x in s])
            if len(assigned_ids) == 1:
                return False
            elif len(assigned_ids) == 2 and None in assigned_ids:
                return False
            else:
                return True

    def handle_overlap_conflict(
        self, to_be_merged: list[set[int]]
    ) -> tuple[list[set[int]], list[int]]:
        """check if there is overlap conflict in to_be_merged. This only happens when merging by appearance

        Args:
            to_be_merged (list[set[int]]): list of ids to be merged together (in self.tracklets)

        Returns:
            list[set[int]]: list of ids to be merged together (in self.tracklets) after handling overlap conflict
            list[int]: list of ids to be removed (in self.tracklets)
        """
        to_be_tmp_removed = [set() for _ in range(self.max_det)]
        to_be_removed = []
        to_be_merged = [list(s) for s in to_be_merged]
        for i, s in enumerate(to_be_merged):
            for p, id1 in enumerate(s):
                for q, id2 in s[p + 1 :]:
                    if detect_overlap(
                        self.tracklets[id1],
                        self.tracklets[id2],
                        self.soft_border,
                        boolean=True,
                    ):
                        if self.tracklets[id1].assigned_id is None:
                            self.tracklets[id1].triggered_conflicts += 1
                            to_be_tmp_removed[i].add(id1)
                            if self.tracklets[id1].triggered_conflicts >= 3:
                                self.logger.warning(
                                    f"removing tracklet {id1} due to overlap conflict"
                                )
                                to_be_removed.append(id1)
                        if self.tracklets[id2].assigned_id is None:
                            self.tracklets[id2].triggered_conflicts += 1
                            to_be_tmp_removed[i].add(id2)
                            if self.tracklets[id2].triggered_conflicts >= 3:
                                self.logger.warning(
                                    f"removing tracklet {id2} due to overlap conflict"
                                )
                                to_be_removed.append(id2)
                # TODO: maybe also remove corresponding intervals from assigned track

        to_be_merged = [
            set(s) - to_be_tmp_removed[i] for i, s in enumerate(to_be_merged)
        ]
        to_be_merged = [s for s in to_be_merged if len(s) > 0]
        return to_be_merged, to_be_removed

    def _merge(
        self,
        to_be_merged: list[set[int]],
        to_be_removed: list[int],
    ):
        """merge tracklets in to_be_merged and remove tracklets in to_be_removed

        Args:
            to_be_merged (list[set[int]]): list of ids to be merged together (in self.tracklets)
            to_be_removed (list[int]): list of ids to be removed (in self.tracklets) and placed in self.dumped
        """
        merged = []
        for s in to_be_merged:
            tracklets = [self.tracklets[x] for x in s]
            tracklets[0] += sum(tracklets[1:])
            merged += s[1:]
            new_tracklets = [x for x in tracklets if x.assigned_id is None]
            self.accumulated_new_frames += sum([len(x) for x in new_tracklets])
        self.remove_tracklets(merged, to_be_removed)

    def predict_identity(self):
        for t in track_iter_progress(self.tracklets):
            ids = t.get_ids()
            fpaths = []
            for id in ids:
                fpaths += [
                    str(self.img_root / str(id) / x)
                    for x in os.listdir(self.img_root / f"{id}")
                ]
            score = self.model.predict(fpaths).mean(axis=0)
            t.pred_score = score

    def _increase_appearance_threshold(self):
        self.apperance_threshold *= 1.5
        self.logger.info(f"setting appearance threshold to {self.apperance_threshold}")

    def _decrease_appearance_threshold(self):
        if self.apperance_threshold > self._appearance_threshold:
            self.apperance_threshold /= 1.5
            self.logger.info(
                f"re-setting appearance threshold to {self.apperance_threshold}"
            )

    def merge_by_appearance(self):
        self.predict_identity()
        conflict_list = self.get_conflict_list(self.soft_border)
        to_be_merged = [set() for _ in range(self.max_det)]
        for t in self.tracklets:
            pred_class, pred_score = t.pred_score.argmax(), t.pred_score.max()
            if t.assigned_id is not None:
                to_be_merged[t.assigned_id].add(t)
                assert (
                    pred_class == t.assigned_id
                ), f"predicted class not equal to assigned class {pred_class} != {t.assigned_id}"
            if pred_score > self.conf_threshold:
                to_be_merged[pred_class].add(t)

        total_length = 0
        for s in to_be_merged:
            total_length += sum([len(x) for x in s])
        if total_length >= self.min_merge_frames:
            self._decrease_appearance_threshold()
            to_be_merged, to_be_removed = self.handle_overlap_conflict(to_be_merged)
            # TODO: 有些是remove是放入dumped，有些是merged，有不同的remove_tracklets处理方式
            self._merge(to_be_merged, to_be_removed)
            return

        self.logger.warning(
            f"No tracklets with score > {self.conf_threshold}. Now merging by appearance"
        )
        cliques, to_be_removed = self.detect_cliques(conflict_list)
        target_scores = np.array([x.pred_score for x in self.assigned_tracklets])
        for c in cliques:
            ids = c["ids"]
            scores = np.array([self.tracklets[id].pred_score for id in ids])
            dist_mat = scipy_distance.cdist(target_scores, scores, metric="cosine")
            matches = match(dist_mat)
            total_dist = dist_mat[matches[:, 0], matches[:, 1]].sum()
            if total_dist / len(ids) < self.apperance_threshold:
                for i, j in matches:
                    to_be_merged[i].add(ids[j])
        total_length = 0
        for s in to_be_merged:
            total_length += sum([len(x) for x in s])
        if total_length < self.min_merge_frames:
            self._increase_appearance_threshold()
        else:
            self._decrease_appearance_threshold()
        self._merge(to_be_merged, to_be_removed)
