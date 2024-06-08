from flask_restx import Api

api = Api(
    title="segTracker.ai API",
    version="1.0",
    description="segTracker.ai API",
    prefix="/api",
    validate=True,
    doc="/api/docs",
)

from .shared_models import api as shared_models_api
from .users.routes import api as users_api
from .videos.routes import api as videos_api
from .jobs.routes import api as jobs_api
from .datasets.routes import api as datasets_api
from .tracks.routes import api as tracks_api
from .models.routes import api as models_api


api.add_namespace(shared_models_api)
api.add_namespace(users_api, path="/users")
api.add_namespace(videos_api, path="/videos")
api.add_namespace(jobs_api, path="/jobs")
api.add_namespace(datasets_api, path="/datasets")
api.add_namespace(tracks_api, path="/tracks")
api.add_namespace(models_api, path="/models")
