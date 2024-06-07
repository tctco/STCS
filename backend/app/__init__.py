import os
from flask import Flask, g, send_from_directory
from flask_cors import CORS
from .common.database import Session, init_db, engine
from pathlib import Path
from .config import SECRET_KEY

resource_path = Path(os.environ.get("RESOURCE_PATH", "../static"))
VIDEO_ROOT_PATH = Path(f"{resource_path}/videos")
IMAGE_ROOT_PATH = Path(f"{resource_path}/images")
EXP_ROOT_PATH = Path(f"{resource_path}/exp")
MODEL_CONFIG_PATH = Path(f"./trained_models/configs")
MODEL_ROOT_PATH = Path(f"{resource_path}/models")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost")
if REDIS_URL.startswith("redis://"):
    REDIS_URL = REDIS_URL[8:]
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)


def create_app():
    from flask_migrate import Migrate
    import rq_dashboard

    app = Flask(__name__, static_url_path="/static", static_folder="static")
    CORS(app, supports_credentials=True)
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["VIDEO_ROOT_PATH"] = VIDEO_ROOT_PATH
    app.config["IMAGE_ROOT_PATH"] = IMAGE_ROOT_PATH
    app.config["EXP_ROOT_PATH"] = EXP_ROOT_PATH
    app.config["MODEL_ROOT_PATH"] = MODEL_ROOT_PATH
    app.config["MODEL_CONFIG_PATH"] = MODEL_CONFIG_PATH
    app.config["RQ_DASHBOARD_REDIS_URL"] = f"redis://{REDIS_URL}:{REDIS_PORT}"
    app.config.from_object(rq_dashboard.default_settings)
    rq_dashboard.web.setup_rq_connection(app)
    app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")

    @app.route("/", methods=["GET"])
    def index():
        return "<h1>Welcome to segTracker.ai!</h1>"

    @app.route("/static/<resource>/<user_id>/<video_token>", methods=["GET"])
    def static_video(resource: str, user_id: int, video_token: str):
        print("sending videos", VIDEO_ROOT_PATH / f"{user_id}/{video_token}")
        # return send_from_directory(VIDEO_ROOT_PATH, f"{user_id}/{video_token}")
        return send_from_directory("../static", f"{resource}/{user_id}/{video_token}")

    from .api import api

    api.init_app(app)

    # Initialize the database
    migrate = Migrate(app, engine, compare_type=True)
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
