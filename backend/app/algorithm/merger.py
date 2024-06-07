from dataclasses import dataclass
from pathlib import Path
import os
import random
import json
import shutil
import logging
from itertools import combinations
from typing import Union
import numpy as np
import networkx as nx
from constants import SOFT_BORDER
from app.algorithm.classifier import MMPretrainClassifier
from app.algorithm.common import match, random_color, set_job_meta
from app.algorithm.models import TrackletStat
import scipy.spatial.distance as scipy_distance
from tqdm import tqdm
import mmcv
import matplotlib.pyplot as plt
from ast import literal_eval


def merge_sets_with_common_elements(sets: list[set]) -> list[set]:
    """merge sets with common elements
    see: https://stackoverflow.com/questions/27180877/how-can-i-merge-lists-that-have-common-elements-within-a-list-of-lists

    Args:
        sets (list[set]): list of sets

    Returns:
        list[set]: list of sets after merging
    """
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
    """Interval of a tracklet"""

    id: int  # original id of the tracklet from the database
    start: int
    end: int  # inclusive


@dataclass
class Clique:
    ids: list[int]
    lifespan: int

    def __len__(self):
        return len(self.ids)


class Tracklet:
    def __init__(self, intervals: list[Interval], assigned_id=None) -> None:
        self.intervals: list[Interval] = intervals
        self.pred_score: np.ndarray = None
        self.assigned_id = assigned_id  # track id assigned by merger, start from 0
        self.triggered_conflicts = 0

    def __repr__(self) -> str:
        return f"Tracklet(id={self.assigned_id}, length={len(self)}, intervals={self.intervals})"

    def remove_by_id(self, id: int):
        self.intervals = [x for x in self.intervals if x.id != id]

    def get_ids(self) -> set[int]:
        return set(x.id for x in self.intervals)

    def get_frames(self) -> list[int]:
        """get frames of the tracklet

        Returns:
            list[int]: list of frames
        """
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
        return sum(x.end - x.start + 1 for x in self.intervals)


def detect_overlap(
    t1: Tracklet,
    t2: Tracklet,
    threshold: int = SOFT_BORDER,
    boolean: bool = False,
) -> Union[bool, list[tuple[int, int]]]:
    """detect overlap between two tracklets

    Args:
        t1 (Tracklet)
        t2 (Tracklet)
        threshold (int, optional): TODO. Defaults to SOFT_BORDER.
        boolean (bool, optional): TODO. Defaults to False.

    Returns:
        Union[bool, list[tuple[int, int]]]: if boolean is True, return True if overlap exists, else return list of tuples of ids that overlap
    """
    i, j = 0, 0
    result = []
    while i < len(t1.intervals) and j < len(t2.intervals):
        if t1.intervals[i].end < t2.intervals[j].start:
            i += 1
        elif t1.intervals[i].start > t2.intervals[j].end:
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
        logger: logging.Logger,
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
        self.tracklets: list[Tracklet] = sorted(
            tracks, key=lambda x: x.intervals[0].start
        )
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
        self.plot_cnt = 0
        self.plot_path = self.exp_root / "plots"
        if self.plot_path.exists():
            shutil.rmtree(self.plot_path)
        self.plot_path.mkdir(exist_ok=True)

        self.max_merge_iterations = 1000

        self.logger.info(f"total # of tracklets: {len(self.tracklets)}")
        self.plot_tracklets("Before Filtering")
        self.filter_tracks()
        self.plot_tracklets("After Filtering")

    def merge(self) -> None:
        self.logger.debug(f"{self.tracklets[0]}\n{self.tracklets[1]}")
        self.init_tracks_kernel()
        self.logger.debug(f"{self.tracklets[1]}")
        self.plot_tracklets("Established initial tracks")
        self.update_cls_model(self.calculate_patience(first=True))
        for iteration in range(self.max_merge_iterations):
            progress = self.get_assigned_length() / self.get_total_length()
            set_job_meta("progress", progress)
            set_job_meta("tracklets", self.get_tracklets_json())
            self.logger.info(
                f"Merged: {self.get_assigned_length()}, Total: {self.get_total_length()}, Ratio: {progress:.2%}"
            )
            if (
                len(self.tracklets) <= self.max_det
                or self.get_assigned_length()
                >= self.get_total_length() * self.train_ratio
            ):
                break
            self.merge_by_appearance()
            self.logger.debug("merged by appearance")
            self.plot_tracklets("merged by appearance")

            self.merge_confirmed_tracklets()
            self.logger.debug("merged by spatio-temporal overlap")
            self.plot_tracklets("merged by spatio-temporal overlap")
            self.logger.info(f"Newly accumulated frames: {self.accumulated_new_frames}")

            self.update_cls_model(patience=self.calculate_patience())

        if iteration == self.max_merge_iterations - 1:
            self.logger.warning(
                f"max merge iterations reached: {self.max_merge_iterations}"
            )
        self.logger.info(
            f"Merged: {self.get_assigned_length()}, Total: {self.get_total_length()}, Ratio: {self.get_assigned_length()/self.get_total_length():.2%}"
        )
        self.logger.info("finished the main merging process.")
        if len(self.tracklets) > self.max_det or len(self.dumped) > 0:
            self.process_dumped()
        self.logger.info(
            f"total train time: {self.model.train_timer.t}\ntotal predict time: {self.model.predict_timer.t}"
        )
        set_job_meta("progress", 1.0)
        set_job_meta("tracklets", self.get_tracklets_json())

    def get_tracklets_json(self):
        result = []
        db_tracklets = self.session.query(TrackletStat).filter(
            TrackletStat.video_id == self.video_id
        )
        idmapper = {}  # db_id -> assigned_id
        for t in self.tracklets:
            if t.assigned_id is not None:
                for i in t.intervals:
                    idmapper[i.id] = t.assigned_id
        for t in db_tracklets:
            result.append(
                {
                    "intervals": literal_eval(t.intervals),
                    "rawTrackID": t.track_id,
                    "trackID": idmapper.get(t.track_id, None),
                }
            )
        return result

    def process_dumped(self):
        self.logger.info("Now processing dumped tracklets...")
        self.tracklets.extend(self.dumped)
        self.plot_tracklets("Dumped added")
        self.tracklets.sort(key=lambda x: x.intervals[0].start)
        self.merge_confirmed_tracklets()
        self.plot_tracklets("Dumped merged by spatio-temporal overlap")
        self.merge_by_appearance(processing_dumped=True)
        self.plot_tracklets("Dumped merged by appearance")
        self.merge_confirmed_tracklets()
        self.plot_tracklets("Final")

    def update_cls_model(self, patience: int):
        train, val = self.generate_dataset()
        with open(self.img_root / "train.json", "w") as f:
            json.dump(train, f)
        with open(self.img_root / "val.json", "w") as f:
            json.dump(val, f)
        self.model.train(patience=patience)

    def filter_tracks(self) -> None:
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
        if len(mean_areas) <= self.max_det * 2:
            self.logger.info("too few tracks, skip track filtering")
            return
        mean_area = np.average(mean_areas, weights=mean_weights)
        std_area = np.sqrt(np.cov(mean_areas, aweights=mean_weights))
        speeds = mean_distance / mean_weights
        mean_speed = np.average(speeds, weights=mean_weights)
        std_speed = np.sqrt(np.cov(speeds, aweights=mean_weights))

        area_threshold = mean_area + 3 * std_area
        area_threshold_low = mean_area - 3 * std_area
        too_large_area = list(
            np.arange(len(self.tracklets))[mean_areas > area_threshold]
        )
        too_small_area = list(
            np.arange(len(self.tracklets))[mean_areas < area_threshold_low]
        )
        too_short = list(
            np.arange(len(self.tracklets))[mean_weights <= self.soft_border]
        )
        speed_threshold = mean_speed - 3 * std_speed
        too_slow = list(np.arange(len(self.tracklets))[speeds < speed_threshold])
        not_confident = list(
            np.arange(len(self.tracklets))[mean_conf < self.min_confidence]
        )

        self.logger.info(f"mean area: {mean_area}, std area: {std_area}")
        self.logger.info(f"mean speed: {mean_speed}, std speed: {std_speed}")
        self.logger.info(f"too large mean area: {len(too_large_area)}")
        self.logger.info(f"too small mean area: {len(too_small_area)}")
        self.logger.info(f"too short: {len(too_short)}")
        self.logger.info(f"too slow: {len(too_slow)}")
        self.logger.info(f"not confident enough: {len(not_confident)}")
        to_be_removed = list(
            set(too_short + not_confident + too_large_area + too_slow + too_small_area)
        )
        self.logger.info(f"total # to be removed: {len(to_be_removed)}")
        self.remove_tracklets([], to_be_removed)

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
        return sum(len(x) for x in self.assigned_tracklets)

    def get_total_length(self) -> int:
        return sum(len(x) for x in self.tracklets)

    def calculate_patience(self, first: bool = False) -> int:
        if first:
            return 5
        patience = 3 * round(
            1 + self.accumulated_new_frames * 8 / self.get_assigned_length()
        )
        return patience

    def init_tracks_kernel(self) -> None:
        """initialize tracklets by spatio-temporal overlap"""
        self.merge_confirmed_tracklets()
        self.plot_tracklets("Before assign id")
        conflict_list = self.get_conflict_list(self.soft_border)
        cliques, to_be_removed = self.detect_cliques(conflict_list)
        assert (
            len(to_be_removed) == 0
        ), f"to_be_removed should be empty, but get {to_be_removed}"
        cliques.sort(key=lambda x: (len(x), x.lifespan), reverse=True)
        c = cliques[0]
        assert (
            len(c) == self.max_det
        ), f"len(c) should be {self.max_det}, but get {c}. Please check if the max_det is correct!"
        self.assigned_tracklets = [None] * self.max_det
        for assigned_id, index in enumerate(c.ids):
            self.tracklets[index].assigned_id = assigned_id
            self.assigned_tracklets[assigned_id] = self.tracklets[index]
        assert None not in self.assigned_tracklets, "None in assigned_tracklets"
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
            self.logger.debug(
                f"tracklet: {t}, ids: {ids}, assigned_id: {t.assigned_id}"
            )
            for id in ids:
                self.logger.debug(f"img root {str(self.img_root / str(id))}")
                fpaths = [
                    str(self.img_root / str(id) / x)
                    for x in os.listdir(self.img_root / f"{id}")
                ]
                split_point = int(len(fpaths) * self.train_ratio)
                self.trained_times[id] = self.trained_times.get(id, 0) + 1
                datalist = [
                    {"img_path": fpath, "gt_label": t.assigned_id} for fpath in fpaths
                ]
                weights = [1 / self.trained_times[id]] * len(datalist)
                if self.trained_times[id] > 1:
                    random.shuffle(datalist)
                train_datalist.extend(datalist[:split_point])
                train_weights.extend(weights[:split_point])

                val_datalist.extend(datalist[split_point:])
                val_weights.extend(weights[split_point:])
            if len(train_datalist) + len(val_datalist) > self.max_frames_per_class:
                train_datalist = random.choices(
                    train_datalist,
                    weights=train_weights,
                    k=int(self.max_frames_per_class * self.train_ratio),
                )
                val_datalist = random.choices(
                    val_datalist,
                    weights=val_weights,
                    k=int(self.max_frames_per_class * (1 - self.train_ratio)),
                )
            elif len(datalist) < self.batch_size:
                self.logger.debug(f"{len(train_datalist)},{len(train_weights)}")
                train_datalist = random.choices(
                    train_datalist, weights=train_weights, k=self.batch_size
                )
                val_datalist = random.choices(
                    val_datalist, weights=val_weights, k=self.batch_size
                )
            self.logger.info(
                f"class: {t.assigned_id}, train: {len(train_datalist)}, val: {len(val_datalist)}"
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
                if detect_overlap(t1, t2, threshold=soft_border, boolean=True):
                    result[i].add(i + j + 1)
                    result[i + j + 1].add(i)
        return result

    def detect_cliques(
        self, conflict_list: list[set[int]]
    ) -> tuple[list[Clique], list[int]]:
        """detect cliques in conflict_list

        Args:
            conflict_list (list[set[int]]): conflict_list[i] is the set of tracklets that conflict with tracklet i

        Returns:
            tuple[list[dict], list[int]]:
                cliques (ordered by minimum length in a clique), {"ids": list[int], "lifespan": int}
                to_be_removed (detected cliques that are longer than max_det)
        """
        graph = nx.Graph()
        for i, s in enumerate(conflict_list):
            graph.add_node(i)
            for j in s:
                graph.add_edge(i, j)
        self.plot_tracklets_network(graph)
        cliques = nx.find_cliques(graph)
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
            lifespan = min(len(self.tracklets[x]) for x in c)
            result.append(Clique(c, lifespan))
        return result, to_be_removed

    def remove_tracklets(self, merged: list[int], to_be_removed: list[int]) -> None:
        """remove tracklets in to_be_removed from self.tracklets

        Args:
            merged (list[int]): list of ids (in self.tracklets) that are merged and should be removed from self.tracklets
            to_be_removed (list[int]): list of ids (in self.tracklets) that are discarded and should be placed in self.dumped
        """
        if len(to_be_removed) > 0:
            original_ids = []
            for t in to_be_removed:
                original_ids += [x.id for x in self.tracklets[t].intervals]
            self.logger.warning(
                f"to be removed is not empty: the original ids are {original_ids}"
            )
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
        # TODO: something not correct
        while self.soft_border < 60 * 10:
            conflict_list = self.get_conflict_list(self.soft_border)
            self.logger.debug(f"conflict list {conflict_list}")
            cliques, to_be_removed = self.detect_cliques(conflict_list)
            cliques = sorted(cliques, key=len, reverse=True)
            self.logger.debug(f"plot cnt ({self.plot_cnt}) cliques {cliques}")
            sets = []
            for c in cliques:
                ids = c.ids
                if len(c) < self.max_det:
                    break
                for id1 in ids:
                    confirmed = set(range(len(self.tracklets))) - conflict_list[id1]
                    for id2 in ids:
                        if id2 == id1:
                            continue
                        confirmed = confirmed.intersection(conflict_list[id2])
                    self.logger.debug(f"clique {c}: confirmed {confirmed}")
                    if (
                        len(confirmed) >= 2
                    ):  # the tracklet itself and the other tracklet
                        assert id1 in confirmed, f"{id1} not in {confirmed}"
                        sets.append(confirmed)
            if len(sets) > 0:
                sets = merge_sets_with_common_elements(sets)
                self.logger.debug(
                    f"spatio-temporal sets to be merged: {sets}\nto be removed: {to_be_removed}"
                )
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
            assigned_ids = set(self.tracklets[x].assigned_id for x in s)
            if len(assigned_ids) == 1:
                continue
            elif len(assigned_ids) == 2 and None in assigned_ids:
                continue
            else:
                return True
        return False

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
                for id2 in s[p + 1 :]:
                    if detect_overlap(
                        self.tracklets[id1],
                        self.tracklets[id2],
                        self.soft_border,
                        boolean=True,
                    ):
                        if self.tracklets[id1].assigned_id is None:
                            self.tracklets[id1].triggered_conflicts += 1
                            self.logger.warning(
                                f"overlap conflict detected in tracklet {id1} and {id2}"
                            )
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
        self.logger.debug(f"merging {to_be_merged}\nremoving {to_be_removed}")
        merged: list[int] = []
        for s in to_be_merged:
            if len(s) <= 1:
                continue
            l: list[int] = sorted(list(s))
            # tracklets = [self.tracklets[x] for x in l]
            # track_kernel = [x for x in tracklets if x.assigned_id is not None]
            # assert (
            #     len(track_kernel) <= 1
            # ), f"len(track_kernel) should be 0 or 1, but get {track_kernel}"
            # track_kernel = track_kernel[0] if len(track_kernel) else tracklets[0]
            # for tid in l:
            #     t = self.tracklets[tid]
            #     if t.assigned_id is None:
            #         track_kernel += t
            #         merged.append(tid)

            tid_tracklets = [(x, self.tracklets[x]) for x in l]
            track_kernel = [x for x in tid_tracklets if x[1].assigned_id is not None]
            assert (
                len(track_kernel) <= 1
            ), f"len(track_kernel) should be 0 or 1, but get {track_kernel}"
            track_kernel_tid, track_kernel = (
                track_kernel[0] if len(track_kernel) else tid_tracklets[0]
            )
            for tid in l:
                t = self.tracklets[tid]
                if tid != track_kernel_tid:
                    track_kernel += t
                    merged.append(tid)

            # for i in range(1, len(tracklets)):
            #     tracklets[0] += tracklets[i]
            # merged += l[1:]
            # new_tracklets = [x for x in tracklets if x.assigned_id is None]
            # self.accumulated_new_frames += sum(len(x) for x in new_tracklets)
        self.accumulated_new_frames += sum(len(self.tracklets[tid]) for tid in merged)
        self.remove_tracklets(merged, to_be_removed)

    def predict_identity(self, max_images: int = 5000):
        for t in tqdm(self.tracklets):
            ids = t.get_ids()
            fpaths = []
            for id in ids:
                fpaths += [
                    str(self.img_root / str(id) / x)
                    for x in os.listdir(self.img_root / f"{id}")
                ]
            if len(fpaths) > max_images:
                fpaths = random.sample(fpaths, max_images)
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

    def merge_by_appearance(self, processing_dumped: bool = False):
        self.predict_identity()
        conflict_list = self.get_conflict_list(self.soft_border)
        to_be_merged = [set() for _ in range(self.max_det)]
        for i, t in enumerate(self.tracklets):
            pred_class, pred_score = t.pred_score.argmax(), t.pred_score.max()
            if t.assigned_id is not None:
                self.logger.info(
                    f"track {t.assigned_id} has predicted score {np.around(t.pred_score,2)}"
                )
                to_be_merged[t.assigned_id].add(i)
                assert (
                    pred_class == t.assigned_id
                ), f"predicted class not equal to assigned class {pred_class} != {t.assigned_id}"
            if pred_score > self.conf_threshold:
                to_be_merged[pred_class].add(i)

        total_length = 0
        for s in to_be_merged:
            total_length += sum(
                len(self.tracklets[x])
                for x in s
                if self.tracklets[x].assigned_id is None
            )
        if total_length >= self.min_merge_frames and not processing_dumped:
            self.logger.info(
                f"Found high confidence tracklets, merging by appearance with conf {self.conf_threshold}. new frames: {total_length}"
            )
            self._decrease_appearance_threshold()
            to_be_merged, to_be_removed = self.handle_overlap_conflict(to_be_merged)
            # TODO: 有些是remove是放入dumped，有些是merged，有不同的remove_tracklets处理方式
            self._merge(to_be_merged, to_be_removed)
            return

        if processing_dumped:
            self.logger.info("merging dumped tracklets by appearance")
        else:
            self.logger.warning(
                f"No tracklets with score > {self.conf_threshold}. Now merging by appearance"
            )
        cliques, to_be_removed = self.detect_cliques(conflict_list)
        target_scores = np.array([x.pred_score for x in self.assigned_tracklets])
        newly_assigned_tracks = {}
        for c in cliques:
            ids = c.ids
            scores = np.array([self.tracklets[id].pred_score for id in ids])
            dist_mat = scipy_distance.cdist(target_scores, scores, metric="cosine")
            matches, *_ = match(dist_mat)
            total_dist = dist_mat[matches[:, 0], matches[:, 1]].sum()
            if total_dist / len(ids) < self.apperance_threshold or processing_dumped:
                for i, j in matches:
                    assigned_tracklet_id = i
                    new_tracklet_id = ids[j]
                    if newly_assigned_tracks.get(new_tracklet_id) in [None, i]:
                        newly_assigned_tracks[new_tracklet_id] = assigned_tracklet_id
                        to_be_merged[assigned_tracklet_id].add(new_tracklet_id)
                    else:  # 一个新tracklet被分配给多个track
                        self.logger.warning(
                            f"conflict clique detected for tracklet {new_tracklet_id} (assigned {newly_assigned_tracks.get(new_tracklet_id)} and {assigned_tracklet_id})"
                        )
                        self.tracklets[new_tracklet_id].triggered_conflicts += 1
                        if self.tracklets[new_tracklet_id].triggered_conflicts >= 3:
                            self.logger.warning(
                                f"removing tracklet {new_tracklet_id} due to conflict"
                            )
                            to_be_removed.append(new_tracklet_id)
                        to_be_merged[newly_assigned_tracks[new_tracklet_id]].remove(
                            new_tracklet_id
                        )
        total_length = 0
        for s in to_be_merged:
            total_length += sum(len(self.tracklets[x]) for x in s)
        if total_length < self.min_merge_frames:
            self._increase_appearance_threshold()
        else:
            self._decrease_appearance_threshold()
        self._merge(to_be_merged, to_be_removed)

    def plot_tracklets(self, title=""):
        """plot tracklets

        Args:
            title (str, optional): title/description of the tracklets plot. Defaults to "".
        """

        fig, ax = plt.subplots(figsize=(12, 4))
        for i, t in enumerate(self.tracklets):
            if t.assigned_id is None:
                color = "gray"
            else:
                color = random_color(t.assigned_id)
                color = (color[0] / 255, color[1] / 255, color[2] / 255)
            for interval in t.intervals:
                ax.plot([interval.start, interval.end], [i, i], color=color)
            ax.text(
                t.intervals[0].start,
                i,
                f"{i}",
                verticalalignment="center",
            )
        if title:
            ax.set_title(title)

        fig.savefig(
            self.plot_path / f"{self.plot_cnt}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")
        fig, ax = plt.subplots(figsize=(12, 4))
        for t in self.assigned_tracklets:
            for interval in t.intervals:
                ax.plot([interval.start, interval.end], [t.assigned_id, t.assigned_id])
            ax.text(
                t.intervals[0].start,
                t.assigned_id,
                f"{t.assigned_id}({t.get_ids()})",
                verticalalignment="center",
            )
        fig.savefig(
            self.plot_path / f"{self.plot_cnt}_assigned.png",
            bbox_inches="tight",
            dpi=300,
        )
        self.plot_cnt += 1

    def plot_tracklets_network(self, graph: nx.Graph, title: str = ""):
        """plot tracklets network

        Args:
            graph (nx.Graph): graph of tracklets
            title (str, optional): title/description of the plot. Defaults to "".
        """
        plt.close("all")
        colors = []
        sizes = []
        for n in graph.nodes:
            sizes.append(len(self.tracklets[n]))
            if self.tracklets[n].assigned_id is None:
                colors.append("gray")
            else:
                color = random_color(self.tracklets[n].assigned_id)
                colors.append((color[0] / 255, color[1] / 255, color[2] / 255))
        nx.draw_networkx(graph, node_color=colors, node_size=sizes, alpha=0.5)
        if title:
            plt.title(title)
        plt.savefig(
            self.plot_path / f"{self.plot_cnt}_network.png",
            bbox_inches="tight",
            dpi=300,
        )
        self.plot_cnt += 1
