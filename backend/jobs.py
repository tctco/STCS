from redis import Redis
from rq import Queue

from dataclasses import dataclass
from typing import Optional
from constants import (
    DET_CHECKPOINT_FILE,
    DET_CONFIG_FILE,
    POSE_CHECKPOINT_FILE,
    POSE_CONFIG_FILE,
    FLOW_CHECKPOINT_FILE,
    FLOW_CONFIG_FILE,
    NMS_THRESH,
    CONF_THRESH,
    MAX_AGE,
    DET_BATCH_SIZE,
    IOU_THRESHOLD,
    MAX_WIDTH_HEIGHT,
    CLS_CONFIG_FILE,
    CLS_CHECKPOINT_FILE,
    BATCH_SIZE,
    SOFT_BORDER,
    ANIMAL_CONFIGS,
    MODELS_CONFIGS,
)
import os


@dataclass
class Params:
    video_name: str
    video_path: str
    max_det: int
    enable_flow: bool
    segm_model_id: str
    pose_model_id: str
    flow_model_id: Optional[str]


@dataclass
class Task:
    priority: int
    params: Params
    video_id: int
    user_id: int
    code: Optional[int]
    animal: str


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost")
if REDIS_URL.startswith("redis://"):
    REDIS_URL = REDIS_URL[8:]
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
redis = Redis(REDIS_URL, REDIS_PORT)
task_queue = Queue(connection=redis)

import subprocess
from sqlalchemy.orm import sessionmaker
import sys
import numpy as np
from ast import literal_eval
from pathlib import Path
import json
import os
from app.common.database import engine

resource_path = os.environ.get("RESOURCE_PATH", "static")


def track(task: Task):
    from app.algorithm.models import Datum, TrackletStat
    from app.api.videos.models import Video
    from app.algorithm.stage1 import online_track
    from app.algorithm.stage2 import merge

    session = sessionmaker(bind=engine)()
    video = session.query(Video).filter(Video.id == task.video_id).first()
    video.analyzed = False
    session.commit()
    det_config_file = MODELS_CONFIGS[task.params.segm_model_id]["config"]
    det_checkpoint_file = MODELS_CONFIGS[task.params.segm_model_id]["checkpoint"]
    pose_config_file = MODELS_CONFIGS[task.params.pose_model_id]["config"]
    pose_checkpoint_file = MODELS_CONFIGS[task.params.pose_model_id]["checkpoint"]
    if task.params.enable_flow:
        flow_config_file = MODELS_CONFIGS[task.params.flow_model_id]["config"]
        flow_checkpoint_file = MODELS_CONFIGS[task.params.flow_model_id]["checkpoint"]
    else:
        flow_config_file = ""
        flow_checkpoint_file = ""

    online_track(
        resource_path,
        task.video_id,
        MAX_WIDTH_HEIGHT,
        task.params.max_det,
        DET_BATCH_SIZE,
        task.params.enable_flow,
        CONF_THRESH,
        NMS_THRESH,
        IOU_THRESHOLD,
        MAX_AGE,
        det_config_file,
        det_checkpoint_file,
        pose_config_file,
        pose_checkpoint_file,
        flow_config_file,
        flow_checkpoint_file,
    )
    # cmd = [
    #     f"{sys.executable}",
    #     "track_stage1.py",
    #     task.params.video_name,
    #     task.params.video_path,
    #     str(task.params.max_det),
    # ]
    # if task.params.enable_flow:
    #     cmd.append("--enable_flow")
    # return_code = subprocess.call(cmd)
    # if return_code != 0:
    #     raise Exception("Error occurred while performing tracking stage 1")

    # merge(
    #     resource_path,
    #     task.video_id,
    #     task.params.max_det,
    #     CLS_CONFIG_FILE,
    #     CLS_CHECKPOINT_FILE,
    #     BATCH_SIZE,
    #     256,
    #     0.6,
    #     SOFT_BORDER,
    # )

    cmd = [
        f"{sys.executable}",
        "track_stage2.py",
        task.params.video_name,
        task.params.video_path,
        str(task.params.max_det),
    ]
    # cmd.extend(["--resume", "28"])
    return_code = subprocess.call(cmd)
    if return_code != 0:
        raise Exception("Error occurred while performing tracking stage 2")
    video.analyzed = True
    session.commit()
    # task.code = return_code
    data = session.query(Datum).filter(
        Datum.video_id == task.video_id,
        Datum.track_id <= task.params.max_det,
        Datum.track_id > 0,
    )
    tracklets = session.query(TrackletStat).filter(
        TrackletStat.video_id == task.video_id,
    )
    keypoints = np.zeros(
        (task.params.max_det, video.frame_cnt, ANIMAL_CONFIGS[task.animal]["nkpts"], 2)
    )
    for d in data:
        keypoints[d.track_id - 1, d.frame] = literal_eval(d.keypoints)
    json_data = {
        "headers": {
            "connections": ANIMAL_CONFIGS[task.animal]["connections"],
            "interval": [0, video.frame_cnt - 1],
            "tracklets": [
                {
                    "start": t.start_frame,
                    "end": t.end_frame,
                    "track_id": t.track_id,
                }
                for t in tracklets
            ],
        },
    }
    json_data["data"] = keypoints.tolist()
    token, ext = os.path.splitext(video.name)
    json_path = Path(f"{resource_path}/json/{task.user_id}/{token}.json")
    if not json_path.parent.exists():
        json_path.parent.mkdir(parents=True)
    with open(json_path, "w") as f:
        json.dump(json_data, f)
    session.close()
    return return_code


import subprocess
import tempfile
import os


def run_ffmpeg_with_progress(input_file, output_file):
    # Create a temporary file to store progress info
    import time

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    # Initialize FFmpeg subprocess with -progress
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-preset",
        "ultrafast",
        "-progress",
        temp_file_path,
        output_file,
    ]
    process = subprocess.Popen(
        cmd, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True
    )

    while True:
        # Read the temporary file to get progress
        time.sleep(0.5)
        with open(temp_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "out_time_ms" in line:
                    out_time_ms = int(line.split("=")[1].strip())
                    print(f"Progress: {out_time_ms / 1000000.0} seconds")

        if process.poll() is not None:
            break

    # Remove the temporary file
    os.remove(temp_file_path)

    process.wait()


if __name__ == "__main__":
    input_file = "/mnt/e/projects/videos/6xWT.MOV"
    output_file = "/mnt/e/projects/videos/testvideos/4mice1_out.mp4"
    run_ffmpeg_with_progress(input_file, output_file)
