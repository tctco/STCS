from flask import g, Blueprint
from .models import Model
from app.common.decorators import token_required
from app.api.users.models import User
from flask_restx import Namespace, fields, Resource
from constants import MODELS_CONFIGS

api = Namespace("Models", description="Models")

model = api.model(
    "Model",
    {
        "id": fields.Integer(required=True, description="Model id"),
        "name": fields.String(
            required=True, description="Model name", attribute="model_name"
        ),
    },
)


# class Models(Resource):
#     @api.marshal_with(model, as_list=True)
#     @token_required
#     def get(self, current_user: User):
#         models = g.session.query(Model).filter(Model.user_id == current_user.id).all()
#         return models
@api.route("/")
class Models(Resource):
    def get(self):
        result = []
        for k, v in MODELS_CONFIGS.items():
            result.append({"id": k, **v})
        return result
