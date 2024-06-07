import os
import json
from flask import g
from flask_restx import Resource, Namespace, fields
from app.common.decorators import token_required
from app.api.users.models import User
from app.api.videos.models import Video
from app.common.messages import error_msg
from app.api.shared_models import empty_input_model, message_model

api = Namespace("TrackResults", description="Track results")

interval_data = api.model(
    "IntervalData",
    {
        "intervals": fields.List(fields.Integer, description="Interval of frames"),
        "rawTrackID": fields.Integer,
        "trackID": fields.Integer(allow_null=True),
    },
)
track_result_header = api.model(
    "TrackResultHeader",
    {
        "connections": fields.List(fields.List(fields.Integer)),
        "interval": fields.List(fields.Integer, description="Interval of frames"),
        "tracklets": fields.List(
            fields.Nested(interval_data), description="Tracklets information"
        ),
    },
)
track_result = api.model(
    "TrackResult",
    {
        "headers": fields.Nested(track_result_header, description="Header information"),
        "data": fields.List(
            fields.List(fields.List(fields.List(fields.Float))),
            description="track x frame x kpts x (x,y)",
        ),
    },
)


@api.route("/<int:video_id>")
@api.doc(params={"video_id": "The video id"})
class TrackResults(Resource):
    @api.response(404, "Video not found", message_model)
    @api.expect(empty_input_model, validate=False)
    @api.marshal_with(track_result)
    @token_required
    def get(self, current_user: User, video_id: int):
        video = (
            g.session.query(Video)
            .filter(Video.id == video_id, Video.user_id == current_user.id)
            .first()
        )
        if not video:
            return error_msg("Video not found"), 404
        token, ext = os.path.splitext(video.name)
        json_path = f"/segtracker/resources/json/{current_user.id}/{token}.json"
        if not os.path.exists(json_path):
            return error_msg("Track results not found"), 404
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
