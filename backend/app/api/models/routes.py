from pathlib import Path
import shutil
from natsort import natsorted
from flask import g, current_app
from flask_restx import Namespace, fields, Resource
from app.common.decorators import token_required
from app.common.messages import success_msg, error_msg
from app.api.users.models import User
from app.api.shared_models import message_model
from constants import MODELS_CONFIGS
from .models import Model

api = Namespace("Models", description="Models")

model_api = api.model(
    "Model",
    {
        "id": fields.String(required=True, description="Model id"),
        "name": fields.String(
            required=True,
            description="Model name",
        ),
        "config": fields.String(required=True, description="Model config file path"),
        "checkpoint": fields.String(
            required=True, description="Model checkpoint file path"
        ),
        "animal": fields.String(required=True, description="Animal type"),
        "type": fields.String(required=True, description="Model type"),
        "method": fields.String(required=True, description="Model method"),
        "owned": fields.Boolean(description="Is owned", default=True),
        "trained": fields.Boolean(description="Is trained (finished)?", default=True),
        "visImages": fields.List(
            fields.String, description="Visualization images", nullable=False
        ),
    },
)

config_node_model = api.model(
    "Config (Node) Model",
    {
        "value": fields.String(required=True, description="Relative path"),
        "title": fields.String(required=True, description="Node name"),
        "selectable": fields.Boolean(description="Is selectable", default=True),
        "children": fields.List(fields.Raw, description="Child nodes"),
    },
)

update_model_status_api = api.model(
    "Update Model Status",
    {
        "trained": fields.Boolean(required=True, description="Is trained (finished)?"),
    },
)


# 创建一个函数来构建递归模型的children部分
def add_children_model(model):
    model["children"] = fields.List(fields.Nested(config_node_model))


add_children_model(config_node_model)


class ModelResource(Resource):
    def get_model_root(self, user_id, model_id):
        return (
            Path(current_app.config["MODEL_ROOT_PATH"]) / str(user_id) / str(model_id)
        )

    def get_checkpoint_config(self, model_root: Path):
        checkpoint, config = None, None
        for model_file in model_root.glob("*.pth"):
            if "best" in model_file.stem:
                checkpoint = model_file
                break
        for mconfig_file in model_root.glob("*.py"):
            config = mconfig_file
            break
        return checkpoint, config

    def find_vis_image_folder(self, model_root: Path):
        vis_image_folder = None

        for path in model_root.rglob("*"):
            if path.is_dir() and path.name == "vis_image":
                vis_image_folder = path
                break
        return vis_image_folder

    def _remove_first_part(self, path: Path):
        path_parts = path.parts
        new_path = Path(*path_parts[2:])  # remove "/segtracker" root part
        return new_path

    def get_model_json(self, model, user_id: int, model_id: int):
        model_root = self.get_model_root(user_id, model_id)
        checkpoint, config = self.get_checkpoint_config(model_root)
        vis_image_folder = self.find_vis_image_folder(model_root)
        if vis_image_folder:
            vis_images = natsorted(
                [str(self._remove_first_part(p)) for p in vis_image_folder.glob("*")],
                reverse=True,
            )
        else:
            vis_images = []
        return {
            "id": model.id,
            "name": model.name,
            "config": str(config),
            "checkpoint": str(checkpoint),
            "animal": model.dataset.animal_name,
            "type": model.category,
            "method": model.method,
            "owned": True,
            "trained": model.trained,
            "visImages": vis_images,
        }


@api.route("/")
class Models(ModelResource):
    @api.marshal_with(model_api, as_list=True)
    @token_required
    def get(self, current_user: User):
        result = []
        models = g.session.query(Model).filter(Model.user_id == current_user.id)
        for model in models:
            result.append(self.get_model_json(model, current_user.id, model.id))
        for mid, values in MODELS_CONFIGS.items():
            result.append(
                {
                    "id": mid,
                    "name": mid,
                    "config": values["config"],
                    "checkpoint": values["checkpoint"],
                    "animal": values["animal"],
                    "type": values["type"],
                    "method": values["method"],
                    "owned": False,
                    "trained": True,
                    "visImages": [],
                }
            )
        return result


@api.route("/configs")
class ModelConfigs(Resource):
    @api.marshal_with(config_node_model)
    def get(self):
        model_config_path: Path = current_app.config["MODEL_CONFIG_PATH"]
        return self._build_tree(model_config_path, model_config_path)

    def _build_tree(self, path: Path, ref_root: Path) -> list:
        tree = []
        for item in path.iterdir():
            relative_path = item.relative_to(ref_root)
            if item.is_dir():
                subtree = self._build_tree(item, ref_root)
                tree.append(
                    {
                        "value": str(relative_path),
                        "title": item.name,
                        "selectable": False,
                        "children": subtree,
                    }
                )
            else:
                tree.append({"value": str(relative_path), "title": item.name})
        return tree


@api.route("/<int:model_id>")
class ModelApi(ModelResource):
    @api.marshal_with(model_api)
    @token_required
    def get(self, current_user: User, model_id: int):
        model = g.session.query(Model).filter(Model.id == model_id).first()
        if model is None:
            return error_msg("Model not found"), 404
        if model.user_id != current_user.id:
            return error_msg("You can only update your own models"), 403
        return self.get_model_json(model, current_user.id, model.id)

    @api.response(403, "Unauthorized delete", message_model)
    @api.response(200, "Model deleted successfully", message_model)
    @token_required
    def delete(self, current_user: User, model_id: int):
        model = g.session.query(Model).get(model_id)
        if model.user_id != current_user.id:
            return error_msg("You can only delete your own models"), 403
        model_root = self.get_model_root(current_user.id, model.id)
        if model_root.exists():
            shutil.rmtree(model_root)
        g.session.delete(model)
        g.session.commit()
        return success_msg("Model deleted successfully"), 200

    @api.response(403, "Unauthorized update", message_model)
    @api.response(200, "Model updated successfully", message_model)
    @api.response(404, "Model files not found", message_model)
    @api.expect(update_model_status_api)
    @token_required
    def post(self, current_user: User, model_id: int):
        new_status = api.payload["trained"]
        model = g.session.query(Model).filter(Model.id == model_id).first()
        if model is None:
            return error_msg("Model not found"), 404
        if model.user_id != current_user.id:
            return error_msg("You can only update your own models"), 403
        model_root = self.get_model_root(current_user.id, model.id)
        checkpoint, config = self.get_checkpoint_config(model_root)
        if config is None or checkpoint is None:
            return error_msg("Model files not found"), 404
        model.trained = new_status
        g.session.commit()
        return success_msg("Model updated successfully"), 200
