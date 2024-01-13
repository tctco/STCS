from flask import Flask, g, send_from_directory
from flask_cors import CORS
from .common.database import Session, init_db
from pathlib import Path
from .config import SECRET_KEY
import os

resource_path = os.environ.get("RESOURCE_PATH", "../static")
VIDEO_ROOT_PATH = Path(f"{resource_path}/videos")


def create_app():
    app = Flask(__name__, static_url_path="/static", static_folder="static")
    CORS(app, supports_credentials=True)
    app.config["SECRET_KEY"] = SECRET_KEY

    @app.route("/", methods=["GET"])
    def index():
        return "<h1>Welcome to segTracker.ai!</h1>"

    @app.route("/static/<resource>/<user_id>/<video_token>", methods=["GET"])
    def static_video(resource: str, user_id: int, video_token: str):
        print("sending videos", VIDEO_ROOT_PATH / f"{user_id}/{video_token}")
        # return send_from_directory(VIDEO_ROOT_PATH, f"{user_id}/{video_token}")
        return send_from_directory("../static", f"{resource}/{user_id}/{video_token}")

    # Import blueprints
    # from .api.users.routes import users_blueprint
    # from .api.videos.routes import videos_blueprint
    # from .api.jobs.routes import jobs_blueprint
    # from .api.datasets.routes import datasets_blueprint
    # from .api.tracks.routes import tracks_blueprint
    # from .api.models.routes import models_blueprint

    # app.register_blueprint(users_blueprint, url_prefix="/api/users")
    # app.register_blueprint(videos_blueprint, url_prefix="/api/videos")
    # app.register_blueprint(jobs_blueprint, url_prefix="/api/jobs")
    # app.register_blueprint(datasets_blueprint, url_prefix="/api/datasets")
    # app.register_blueprint(tracks_blueprint, url_prefix="/api/tracks")
    # app.register_blueprint(models_blueprint, url_prefix="/api/models")
    from .api import api

    api.init_app(app)

    # Initialize the database
    init_db()
    from jobs import task_queue
    from rq.command import send_stop_job_command

    jids = task_queue.started_job_registry.get_job_ids()
    for jid in jids:
        send_stop_job_command(task_queue.connection, jid)

    @app.before_request
    def create_session():
        """Create a session before each request."""
        g.session = Session()

    @app.teardown_appcontext
    def close_session(exception=None):
        """Close the session at the end of each request."""
        Session.remove()

    return app
