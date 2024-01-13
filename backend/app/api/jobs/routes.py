from flask import g, request
from app.api.users.models import User
from app.common.messages import success_msg, error_msg
from app.common.decorators import token_required
from app.api.videos.models import Video
from rq.command import send_stop_job_command
from rq.job import Job
from rq.exceptions import NoSuchJobError
import rq
import os
from jobs import Task, Params, track, task_queue, redis
from flask_restx import Resource, Namespace, fields
from app.api.shared_models import empty_input_model, message_model

from pathlib import Path

resource_path = os.environ.get("RESOURCE_PATH", "static")
VIDEO_ROOT_PATH = Path(f"{resource_path}/videos")
job_queue = rq.Queue(connection=redis)


api = Namespace("Jobs", description="Jobs")

job_info = api.model(
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
    },
)


@api.route("/")
class JobResource(Resource):
    @api.marshal_with(job_info, as_list=True)
    @api.expect(empty_input_model, validate=False)
    @token_required
    def get(self, current_user: User):
        job_ids = job_queue.get_job_ids()
        job_ids += job_queue.started_job_registry.get_job_ids()
        job_ids += job_queue.finished_job_registry.get_job_ids()
        job_ids += job_queue.failed_job_registry.get_job_ids()
        job_ids += job_queue.deferred_job_registry.get_job_ids()
        job_ids += job_queue.scheduled_job_registry.get_job_ids()
        job_ids += job_queue.canceled_job_registry.get_job_ids()
        jobs = []
        for jid in reversed(job_ids):
            try:
                job = Job.fetch(jid, connection=redis)
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
        video_id = request.json.get("videoId", None)
        max_det = request.json.get("maxDet", None)
        enable_flow = request.json.get("flow", True)
        animal = request.json.get("animal", None)
        segm_model_id = request.json.get("segmModel", None)
        pose_model_id = request.json.get("poseModel", None)
        flow_model_id = request.json.get("flowModel", None)
        if enable_flow and flow_model_id is None:
            return error_msg("Flow model id is required if enable flow"), 400
        video = (
            g.session.query(Video)
            .filter(Video.id == video_id, Video.user_id == current_user.id)
            .first()
        )
        if not video:
            return error_msg(f"Cannot find video with given video id in the DB"), 400
        vpath = VIDEO_ROOT_PATH / str(current_user.id)
        print(vpath, video.name, os.getcwd())
        if not (vpath / video.name).exists():
            return error_msg(f"[Internal Error] Cannot find video in file system"), 400
        # for t in tasks_list:
        #     if t.video_id == video_id:
        #         return error_msg("This video is already in queue"), 400
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
        job = task_queue.enqueue(
            track,
            Task(
                current_user.priority, params, video.id, current_user.id, None, animal
            ),
            result_ttl=86400,
            job_timeout=86400 * 2,
        )
        return success_msg(f"Job for {video.original_name} added!")


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
            return error_msg(f"No such job {jid}"), 400
        if job.args[0].user_id != current_user.id:
            return error_msg(f"Not authorized"), 401
        if job.get_status() == "started":
            send_stop_job_command(redis, jid)
            return success_msg("job terminated")
        elif job.get_status() == "queued":
            job.cancel()
            return success_msg("job canceled")
