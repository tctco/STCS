from flask import g, url_for
from app.common.decorators import token_required
from flask_restx import Resource, Namespace, fields
from app.api.users.models import User
from app.api.videos.models import Video
import os
from app.common.messages import error_msg
from app.api.shared_models import empty_input_model, message_model

resource_path = os.environ.get("RESOURCE_PATH", "static")
if resource_path == "static":
    resource_path_url = "/static"
else:
    resource_path_url = "/resources"

api = Namespace("TrackResults", description="Track results")

track_result = api.model(
    "TrackResult",
    {
        "path": fields.String(description="Path to the track result file"),
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
            error_msg("Video not found"), 404
        token, ext = os.path.splitext(video.name)
        json_path = f"{resource_path_url}/json/{current_user.id}/{token}.json"
        return {"path": json_path}
