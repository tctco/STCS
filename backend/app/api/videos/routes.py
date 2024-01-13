from flask import request, g, url_for
import os
import shutil
from pathlib import Path
from typing import Tuple
import mmcv

from app.api.users.models import User
from .models import Video
from app.common.messages import success_msg, error_msg
from app.common.decorators import token_required
from werkzeug.utils import secure_filename
import secrets
from flask_restx import Resource, Namespace, fields
from app.api.shared_models import empty_input_model, message_model

resource_path = os.environ.get("RESOURCE_PATH", "./static")
VIDEO_ROOT_PATH = Path(f"{resource_path}/videos")
EXP_FILES_PATH = Path("exp/files")
CLS_MODELS_PATH = Path("exp/cls_models")
resource_path = Path(resource_path)


ALLOWED_EXTENSIONS = set([".mp4"])
CHROME_SUPPORTED_CODEC = [
    "VP80",
    "VP90",
    "av01",
    "H264",
    "h264",
    "X264",
    "x264",
]


def allowed_extension(fname: str) -> Tuple[bool, str]:
    allowed = True
    if Path(fname).suffix not in ALLOWED_EXTENSIONS:
        allowed = False
    return allowed, Path(fname).suffix


def allowed_codec(codec: str) -> bool:
    return codec in CHROME_SUPPORTED_CODEC


def error_msg(msg: str) -> dict:
    return {"success": False, "message": msg}


def success_msg(msg: str) -> dict:
    return {"success": True, "message": msg}


def fourcc2codec(fourcc: int) -> str:
    return (
        chr(fourcc & 0xFF)
        + chr((fourcc >> 8) & 0xFF)
        + chr((fourcc >> 16) & 0xFF)
        + chr((fourcc >> 24) & 0xFF)
    )


api = Namespace("Videos", description="Videos")

video_info = api.model(
    "VideoInfo",
    {
        "key": fields.Integer(required=True, description="Video id"),
        "id": fields.Integer(required=True, description="Video id"),
        "name": fields.String(required=True, description="Video name"),
        "analyzed": fields.Boolean(required=True, description="Video analyzed"),
        "frameCnt": fields.Integer(required=True, description="Video frame count"),
        "fps": fields.Integer(required=True, description="Video fps"),
        "width": fields.Integer(required=True, description="Video width"),
        "height": fields.Integer(required=True, description="Video height"),
        "timeUploaded": fields.DateTime(required=True, description="Video upload time"),
        "size": fields.Integer(required=True, description="Video size"),
        "url": fields.String(required=True, description="Video url"),
    },
)


@api.route("/")
class VideoListResource(Resource):
    @api.expect(empty_input_model, validate=False)
    @api.marshal_with(video_info, as_list=True)
    @token_required
    def get(self, current_user: User):
        videos = g.session.query(Video).filter(Video.user_id == current_user.id)
        videos = [
            {
                "key": v.id,
                "id": v.id,
                "name": v.original_name,
                "analyzed": v.analyzed,
                "frameCnt": v.frame_cnt,
                "fps": v.fps,
                "width": v.width,
                "height": v.height,
                "timeUploaded": v.time_uploaded,
                "size": v.size,
                "url": url_for("resources", f"{current_user.id}/{v.name}"),
            }
            for v in videos
        ]
        return videos

    @api.response(415, "Invalid extension/codec", message_model)
    @api.expect(empty_input_model, validate=False)
    @api.marshal_with(message_model)
    @token_required
    def put(self, current_user: User):
        uploaded_video = request.files["video"]
        original_video_name = secure_filename(uploaded_video.filename)
        video_token = secrets.token_hex(8)
        allowed, ext = allowed_extension(original_video_name)
        if not allowed:
            return (
                error_msg(
                    f"Invalid extension: {ext}. We only support {ALLOWED_EXTENSIONS}."
                ),
                415,
            )
        video_name = f"{video_token}{ext.lower()}"
        path = VIDEO_ROOT_PATH / f"{current_user.id}/{video_name}"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        uploaded_video.save(path)
        video = mmcv.VideoReader(str(path))
        codec = fourcc2codec(int(video.fourcc))
        if not allowed_codec(codec):
            os.remove(path)
            return (
                error_msg(
                    f"Invalid codec: {codec}. We only support {CHROME_SUPPORTED_CODEC}"
                ),
                415,
            )
        size = os.path.getsize(path)
        new_video = Video(
            video_name,
            current_user.id,
            original_video_name,
            video.frame_cnt,
            video.fps,
            video.width,
            video.height,
            size,
        )
        g.session.add(new_video)
        g.session.commit()
        return success_msg(f"{original_video_name} saved successfully!")


video_deletes = api.model(
    "VideoDeletes",
    {
        "videos": fields.List(
            fields.Integer, required=True, description="List of video ids to be deleted"
        )
    },
)


# @api.route("/delete")
# class VideoResource(Resource):
#     @api.response(404, "Video not found", message_model)
#     @api.expect(video_deletes)
#     @api.marshal_with(message_model)
#     @token_required
#     def delete_videos(current_user: User, video_deletes: dict):
#         to_be_deleted = video_deletes["videos"]
#         for video_id in to_be_deleted:
#             video = (
#                 g.session.query(Video)
#                 .filter(Video.id == video_id, Video.user_id == current_user.id)
#                 .first()
#             )
#             if not video:
#                 return error_msg(f"Cannot find video with id {video_id}"), 404
#             g.session.delete(video)
#             v_path = VIDEO_ROOT_PATH / f"{current_user.id}/{video.name}"
#             if v_path.exists():
#                 os.remove(v_path)
#             g.session.commit()
#             purename, ext = os.path.splitext(video.name)
#             json_path = Path(f"{resource_path}/json/{current_user.id}/{purename}.json")
#             if json_path.exists():
#                 os.remove(json_path)
#             cls_models = CLS_MODELS_PATH / purename
#             if cls_models.exists():
#                 shutil.rmtree(cls_models)
#             exp_files = EXP_FILES_PATH / purename
#             if exp_files.exists():
#                 shutil.rmtree(exp_files)
#         return success_msg(f"Successfully deleted videos")


@api.route("/<video_id>")
@api.doc(params={"video_id": "The video id"})
class VideoResource(Resource):
    @api.response(404, "Video not found", message_model)
    @api.marshal_with(message_model)
    @token_required
    def delete(self, current_user: User, video_id: int):
        video = (
            g.session.query(Video)
            .filter(Video.id == video_id, Video.user_id == current_user.id)
            .first()
        )
        if not video:
            return error_msg(f"Cannot find video with id {video_id}"), 404
        g.session.delete(video)
        v_path = resource_path / "videos" / f"{current_user.id}/{video.name}"
        if v_path.exists():
            os.remove(v_path)
        g.session.commit()
        purename, ext = os.path.splitext(video.name)
        json_path = Path(f"{resource_path}/json/{current_user.id}/{purename}.json")
        if json_path.exists():
            os.remove(json_path)
        pure_original_name, ext = os.path.splitext(video.original_name)
        exp_files = (
            resource_path
            / "exp"
            / str(video.user_id)
            / f"{purename}_{pure_original_name}"
        )
        if exp_files.exists():
            shutil.rmtree(exp_files)

        cls_models = CLS_MODELS_PATH / f"cls_{purename}_work_dirs"
        if cls_models.exists():
            shutil.rmtree(cls_models)
        return success_msg(f"Successfully deleted videos")
