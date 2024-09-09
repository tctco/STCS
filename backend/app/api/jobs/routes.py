import os
from pathlib import Path
from typing import Union
from ast import literal_eval
from flask import g, request, current_app
from app.api.users.models import User
from app.common.messages import success_msg, error_msg
from app.common.decorators import token_required
from app.api.videos.models import Video
from app.api.datasets.models import Dataset, Image, Annotation
from app.api.models.models import Model
from app.api.shared_models import empty_input_model, message_model
from rq.command import send_stop_job_command
from rq.job import Job
from rq.exceptions import NoSuchJobError
import rq
from jobs import (
    TrackTask,
    TrainDetTask,
    TrainPoseTask,
    Params,
    track,
    redis,
    train_det,
    train_pose,
)
from flask_restx import Resource, Namespace, fields
from sqlalchemy import func


resource_path = os.environ.get("RESOURCE_PATH", "static")
VIDEO_ROOT_PATH = Path(f"{resource_path}/videos")
job_queue = rq.Queue(connection=redis)


api = Namespace("Jobs", description="Jobs")

track_job_info = api.model(
    "JobInfo",
    {
        "taskId": fields.String(required=True, description="Job id"),
        "trueName": fields.String(required=True, description="User true name"),
        "maxDet": fields.Integer(required=True, description="Max animal det"),
        "videoName": fields.String(required=True, description="Video name (token)"),
        "status": fields.String(required=True, description="Job status"),
        "created": fields.DateTime(required=True, description="Job created time"),
        "started": fields.DateTime(required=True, description="Job started time"),
        "ended": fields.DateTime(required=True, description="Job ended time"),
        "priority": fields.Integer(required=True, description="Job/creator priority"),
        "totalFrames": fields.Integer(required=True, description="Total frames"),
        "owned": fields.Boolean(required=True, description="Owned by current user"),
        "progress": fields.Float(required=False, description="Job progress 0-1"),
    },
)

new_job_request = api.model(
    "NewJobRequest",
    {
        "videoId": fields.Integer(required=True, description="Video id"),
        "maxDet": fields.Integer(required=True, description="Max animal det", min=1),
        "flow": fields.Boolean(required=True, description="Enable flow"),
        "animal": fields.String(required=True, description="Animal type"),
        "segmModel": fields.String(required=True, description="Segm model id"),
        "poseModel": fields.String(required=True, description="Pose model id"),
        "flowModel": fields.String(allow_null=True, description="Flow model id"),
        "baseVideoId": fields.String(
            allow_null=True, description="Base ID model (video id)"
        ),
        "maxTrainingFrames": fields.Integer(
            allow_null=True, description="Max training frames when building ID model"
        ),
    },
)


@api.route("/<jid>")
@api.doc(params={"jid": "The job id (str)"})
class JobCancel(Resource):
    @api.response(404, "Job not found", message_model)
    @api.response(401, "Not authorized", message_model)
    @api.marshal_with(message_model)
    @token_required
    def delete(self, current_user: User, jid: str):
        try:
            job = Job.fetch(jid, connection=redis)
        except NoSuchJobError:
            return error_msg(f"No such job {jid}"), 404
        if job.args[0].user_id != current_user.id:
            return error_msg("Not authorized"), 401
        if job.get_status() == "started":
            send_stop_job_command(redis, jid)
            return success_msg("job terminated")
        elif job.get_status() == "queued":
            job.cancel()
            return success_msg("job canceled")


def fetch_all_job_ids():
    job_ids = job_queue.get_job_ids()
    job_ids += job_queue.started_job_registry.get_job_ids()
    job_ids += job_queue.finished_job_registry.get_job_ids()
    job_ids += job_queue.failed_job_registry.get_job_ids()
    job_ids += job_queue.deferred_job_registry.get_job_ids()
    job_ids += job_queue.scheduled_job_registry.get_job_ids()
    job_ids += job_queue.canceled_job_registry.get_job_ids()
    return job_ids


@api.route("/track")
class TrackJobResource(Resource):
    @api.marshal_with(track_job_info, as_list=True)
    @api.expect(empty_input_model, validate=False)
    @token_required
    def get(self, current_user: User):
        job_ids = fetch_all_job_ids()
        jobs = []
        for jid in reversed(job_ids):
            try:
                job = Job.fetch(jid, connection=redis)
                if not isinstance(job.args[0], TrackTask):
                    continue
                user: User = (
                    g.session.query(User).filter(User.id == job.args[0].user_id).first()
                )
                video: Video = (
                    g.session.query(Video)
                    .filter(Video.id == job.args[0].video_id)
                    .first()
                )
                if video is None:
                    continue
                meta = job.get_meta(refresh=True)
                jobs.append(
                    {
                        "taskId": job.id,
                        "trueName": user.true_name,
                        "maxDet": job.args[0].params.max_det,
                        "videoName": job.args[0].params.video_name,
                        "status": job.get_status(),
                        "created": job.enqueued_at,
                        "started": job.started_at,
                        "ended": job.ended_at,
                        "priority": job.args[0].priority,
                        "totalFrames": video.frame_cnt,
                        "owned": job.args[0].user_id == current_user.id,
                        "progress": meta.get("progress", None),
                        # "meta": job.get_meta(refresh=True),
                    }
                )
            except NoSuchJobError:
                pass
        jobs = sorted(jobs, key=lambda x: x["created"], reverse=True)
        return jobs

    @api.response(400, "Video not found", message_model)
    @api.marshal_with(message_model)
    @api.expect(new_job_request)
    @token_required
    def put(self, current_user: User):
        payload = request.json
        video_id = payload.get("videoId", None)
        max_det = payload.get("maxDet", None)
        enable_flow = payload.get("flow", True)
        animal = payload.get("animal", None)
        segm_model_id = payload.get("segmModel", None)
        pose_model_id = payload.get("poseModel", None)
        flow_model_id = payload.get("flowModel", None)
        base_video_id = payload.get("baseVideoId", None)
        max_training_frames = payload.get("maxTrainingFrames", None)
        if enable_flow and flow_model_id is None:
            return error_msg("Flow model id is required if enable flow"), 400
        video = (
            g.session.query(Video)
            .filter(Video.id == video_id, Video.user_id == current_user.id)
            .first()
        )
        if not video:
            return error_msg("Cannot find video with given video id in the DB"), 400
        vpath = VIDEO_ROOT_PATH / str(current_user.id)
        print(vpath, video.name, os.getcwd())
        if not (vpath / video.name).exists():
            return error_msg("[Internal Error] Cannot find video in file system"), 400
        started_jids = job_queue.started_job_registry.get_job_ids()
        for jid in started_jids:
            job = Job.fetch(jid, connection=redis)
            if job.args[0].video_id == video_id:
                return error_msg("This video is already in queue"), 400
        purename, ext = os.path.splitext(video.name)
        params = Params(
            purename,
            str(vpath),
            max_det,
            enable_flow,
            segm_model_id,
            pose_model_id,
            flow_model_id,
        )
        print("task params is", params)
        job = job_queue.enqueue(
            track,
            TrackTask(
                current_user.priority,
                params,
                video.id,
                current_user.id,
                None,
                animal,
                base_video_id,
                max_training_frames,
            ),
            result_ttl=86400,
            job_timeout=86400 * 2,
        )
        return success_msg(f"Job for {video.original_name} added!")


tracklet_model = api.model(
    "Tracklet",
    {
        "intervals": fields.List(
            fields.Integer, required=True, description="List of interval frame numbers"
        ),
        "rawTrackID": fields.Integer(required=True, description="Raw track ID"),
        "trackID": fields.Integer(
            required=False, description="Mapped track ID", allow_null=True
        ),
    },
)

track_job_meta = api.model(
    "JobMeta",
    {
        "stage": fields.Integer(
            required=True, description="Algorithm stage", allow_null=True
        ),
        "progress": fields.Float(
            required=True, description="Stage process (0-1)", allow_null=True
        ),
        "maxDet": fields.Integer(required=True, description="Max animal det"),
        "tracklets": fields.List(
            fields.Nested(tracklet_model),
            required=True,
            description="Tracklet info",
            allow_null=True,
        ),
    },
)


@api.route("/track/meta/<jid>")
@api.doc(params={"jid": "The job id (str)"})
class TrackJobMeta(Resource):
    @api.response(404, "Job not found", message_model)
    @api.response(401, "Not authorized", message_model)
    @api.response(200, "Track Job meta", track_job_meta)
    @token_required
    def get(self, current_user: User, jid: str):
        try:
            job = Job.fetch(jid, connection=redis)
        except NoSuchJobError:
            return error_msg(f"No such job {jid}"), 404
        if job.args[0].user_id != current_user.id:
            return error_msg("Not authorized"), 401
        meta = job.get_meta(refresh=True)
        if meta.get("tracklets") is None:
            meta["tracklets"] = None
        if meta.get("stage") is None:
            meta["stage"] = None
        if meta.get("progress") is None:
            meta["progress"] = None
        meta["maxDet"] = job.args[0].params.max_det

        return meta, 200


train_det_job_params = api.model(
    "TrainJobParams",
    {
        "config": fields.String(required=True, description="Model config file path"),
        "datasetId": fields.Integer(required=True, description="Dataset id"),
        "valRatio": fields.Float(
            required=True, description="Validation ratio, should be in 0.1 - 0.5"
        ),
        "modelName": fields.String(required=True, description="Model name"),
    },
)


class BaseTrainModelJobResource(Resource):
    def check_payload(self, payload: dict, current_user: User):
        dataset_id = payload["datasetId"]
        dataset = (
            g.session.query(Dataset)
            .filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
            .first()
        )
        if dataset is None:
            return error_msg("Cannot find dataset with given dataset id in the DB"), 404
        img_cnt = (
            g.session.query(func.count(Image.id))
            .join(Annotation)
            .filter(Image.dataset_id == dataset_id)
            .scalar()
        )
        if img_cnt < 10:
            return error_msg("Dataset should have at least 10 images"), 400
        return dataset, 200

    def fetch_job_info(
        self, filter_instance: Union[TrainPoseTask, TrainDetTask], current_user
    ):
        job_ids = fetch_all_job_ids()
        jobs = []
        for jid in reversed(job_ids):
            job = Job.fetch(jid, connection=redis)
            task = job.args[0]
            if not isinstance(task, filter_instance):
                # print("not instance", task)
                print(type(task), filter_instance)
                continue
            user = g.session.query(User).filter(User.id == task.user_id).first()
            dataset = (
                g.session.query(Dataset).filter(Dataset.id == task.dataset_id).first()
            )
            model = g.session.query(Model).filter(Model.id == task.model_id).first()
            if model is None or dataset is None or user is None:
                print(f"model:{model}\ndataset:{dataset}\nuser:{user}\ntask{task}")
                continue
            jobs.append(
                {
                    "taskId": job.id,
                    "trueName": user.true_name,
                    "datasetName": dataset.name,
                    "status": job.get_status(),
                    "created": job.enqueued_at,
                    "started": job.started_at,
                    "ended": job.ended_at,
                    "priority": task.priority,
                    "owned": task.user_id == current_user.id,
                    "config": str(task.config_path),
                    "valRatio": task.val_ratio,
                    "modelName": model.name,
                }
            )
        jobs = sorted(jobs, key=lambda x: x["created"], reverse=True)
        return jobs


det_job_info = api.model(
    "JobInfo",
    {
        "taskId": fields.String(required=True, description="Job id"),
        "trueName": fields.String(required=True, description="User true name"),
        "datasetName": fields.String(required=True, description="Dataset name"),
        "status": fields.String(required=True, description="Job status"),
        "created": fields.DateTime(required=True, description="Job created time"),
        "started": fields.DateTime(required=True, description="Job started time"),
        "ended": fields.DateTime(required=True, description="Job ended time"),
        "priority": fields.Integer(required=True, description="Job/creator priority"),
        "owned": fields.Boolean(required=True, description="Owned by current user"),
        "config": fields.String(required=True, description="Model config file path"),
        "valRatio": fields.Float(required=True, description="Validation ratio"),
        "modelName": fields.String(required=True, description="Model name"),
    },
)


@api.route("/det")
class TrainDetJobResource(BaseTrainModelJobResource):

    @api.marshal_with(det_job_info, as_list=True)
    @token_required
    def get(self, current_user: User):
        jobs = self.fetch_job_info(TrainDetTask, current_user)
        return jobs

    @api.expect(train_det_job_params)
    @api.marshal_with(message_model)
    @token_required
    def post(self, current_user: User):
        payload = request.json
        print("payload", payload)
        check_result, return_code = self.check_payload(payload, current_user)
        if return_code != 200:
            return check_result, return_code
        dataset = check_result
        config = payload["config"]
        dataset_id = payload["datasetId"]
        val_ratio = payload["valRatio"]
        model_name = payload["modelName"]
        IMAGE_ROOT_PATH = Path(current_app.config["IMAGE_ROOT_PATH"])
        dataset_root = IMAGE_ROOT_PATH / str(current_user.id) / str(dataset_id)
        dataset_root = dataset_root.resolve()

        params = {"val_ratio": val_ratio}
        new_model = Model(
            model_name, dataset_id, current_user.id, "segm", config, False, str(params)
        )
        g.session.add(new_model)
        g.session.commit()
        task: TrainDetTask = TrainDetTask(
            current_user.priority,
            config,
            dataset_id,
            current_user.id,
            val_ratio,
            dataset.animal_name,
            dataset_root,
            new_model.id,
        )
        job_queue.enqueue(train_det, task, result_ttl=86400, job_timeout=86400 * 2)
        return success_msg(f"Job for {model_name} added!")


train_pose_job_params = api.model(
    "TrainJobParams",
    {
        "config": fields.String(required=True, description="Model config file path"),
        "datasetId": fields.Integer(required=True, description="Dataset id"),
        "valRatio": fields.Float(
            required=True, description="Validation ratio, should be in 0.1 - 0.5"
        ),
        "modelName": fields.String(required=True, description="Model name"),
        "links": fields.List(
            fields.List(fields.String), required=True, description="keypoint Links"
        ),
        "swaps": fields.List(
            fields.List(fields.String), required=True, description="keypoint Swaps"
        ),
    },
)

pose_job_info = api.model(
    "JobInfo",
    {
        "taskId": fields.String(required=True, description="Job id"),
        "trueName": fields.String(required=True, description="User true name"),
        "datasetName": fields.String(required=True, description="Dataset name"),
        "status": fields.String(required=True, description="Job status"),
        "created": fields.DateTime(required=True, description="Job created time"),
        "started": fields.DateTime(required=True, description="Job started time"),
        "ended": fields.DateTime(required=True, description="Job ended time"),
        "priority": fields.Integer(required=True, description="Job/creator priority"),
        "owned": fields.Boolean(required=True, description="Owned by current user"),
        "config": fields.String(required=True, description="Model config file path"),
        "valRatio": fields.Float(required=True, description="Validation ratio"),
        "modelName": fields.String(required=True, description="Model name"),
        "swaps": fields.List(
            fields.List(fields.String), required=True, description="keypoint Swaps"
        ),
        "links": fields.List(
            fields.List(fields.String), required=True, description="keypoint Links"
        ),
    },
)


@api.route("/pose")
class TrainPoseJobResource(BaseTrainModelJobResource):
    @api.marshal_with(pose_job_info, as_list=True)
    @token_required
    def get(self, current_user: User):
        jobs = self.fetch_job_info(TrainPoseTask, current_user)
        for job_info in jobs:
            jid = job_info["taskId"]
            job = Job.fetch(jid, connection=redis)
            task = job.args[0]
            job_info["links"] = task.skeleton
            job_info["swaps"] = task.swap

        return jobs

    @api.expect(train_det_job_params)
    @token_required
    def post(self, current_user: User):
        payload = request.json
        check_result, return_code = self.check_payload(payload, current_user)
        if return_code != 200:
            return check_result, return_code
        dataset = check_result
        config = payload["config"]
        dataset_id = payload["datasetId"]
        val_ratio = payload["valRatio"]
        model_name = payload["modelName"]
        IMAGE_ROOT_PATH = Path(current_app.config["IMAGE_ROOT_PATH"])
        dataset_root = IMAGE_ROOT_PATH / str(current_user.id) / str(dataset_id)
        dataset_root = dataset_root.resolve()
        params = {
            "link": payload["links"],
            "swap": payload["swaps"],
            "keypoints": literal_eval(dataset.keypoints),
        }
        new_model = Model(
            model_name, dataset_id, current_user.id, "pose", config, False, str(params)
        )
        g.session.add(new_model)
        g.session.commit()
        task: TrainPoseTask = TrainPoseTask(
            current_user.priority,
            Path(config),
            dataset_id,
            dataset_root,
            current_user.id,
            val_ratio,
            dataset.animal_name,
            payload["links"],
            literal_eval(dataset.keypoints),
            payload["swaps"],
            new_model.id,
        )
        job_queue.enqueue(train_pose, task, result_ttl=86400, job_timeout=86400 * 2)
        return success_msg(f"Job for {model_name} added!")
