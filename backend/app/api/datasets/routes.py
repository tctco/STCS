from flask import Blueprint, request, g
from app.common.decorators import token_required
from app.api.users.models import User
from app.common.messages import error_msg, success_msg
from .models import Dataset, Annotation, Image
import numpy as np
from sqlalchemy import func
from ast import literal_eval
from flask_restx import Resource, Namespace, fields
from app.api.shared_models import empty_input_model, message_model

api = Namespace("Datasets", description="Datasets")

# @datasets_blueprint.route("/datasets/update", methods=["POST"])
# def append_dataset():
#     payload = request.json
#     dataset_id = payload.get("datasetId", None)
#     if not dataset_id:
#         return error_msg("Dataset not provided")
#     dataset = g.session.query(Dataset).filter(Dataset.id == dataset_id).first()
#     if dataset is None:
#         return error_msg("cannot find dataset")
#     data = payload.get("data", {})
#     updated, inserted, deleted = 0, 0, 0
#     for annotation_id, payload in data.items():
#         ann = (
#             g.session.query(Annotation).filter((Annotation.id == annotation_id)).first()
#         )

#         kpts = np.array(payload["keypoints"], dtype=int).reshape((-1, 3))
#         xmax, ymax, _ = kpts.max(axis=0)
#         xmin, ymin, _ = kpts.min(axis=0)
#         area = (xmax - xmin) * (ymax - ymin)
#         bbox = [xmin, ymin, xmax - xmin, ymin - ymax]
#         img_id = payload["imageId"]
#         if ann is None and kpts.sum() <= 0:
#             continue
#         elif ann is None and kpts.sum() > 0:
#             ann = Annotation(
#                 image_id=img_id, keypoints=str(kpts), area=area, bbox=str(bbox)
#             )
#             g.session.add(ann)
#             inserted += 1
#         elif ann is not None and kpts.sum() <= 0:
#             g.session.delete(ann)
#             deleted += 1
#         else:
#             ann.keypoints = str(kpts)
#             ann.area = area
#             ann.bbox = str(bbox)
#             updated += 1
#     g.session.commit()
#     return success_msg(
#         f"successfully updated: {updated}, inserted: {inserted}, deleted: {deleted}"
#     )


dataset_info = api.model(
    "DatasetInfo",
    {
        "name": fields.String(required=True, description="Dataset name"),
        "keypoints": fields.List(
            fields.List(fields.String), required=True, description="Keypoints"
        ),
        "created": fields.DateTime(required=True, description="Created time"),
        "images": fields.Integer(required=True, description="Number of images"),
        "annotations": fields.Integer(
            required=True, description="Number of annotations"
        ),
    },
)

dataset_new = api.model(
    "DatasetNew",
    {
        "name": fields.String(required=True, description="Dataset name"),
        "keypoints": fields.List(
            fields.List(fields.String), required=True, description="Keypoints"
        ),
        "animal_name": fields.String(required=True, description="Animal name"),
    },
)


@api.route("/")
class DatasetResource(Resource):
    @api.expect(empty_input_model, validate=False)
    @api.marshal_with(dataset_info, as_list=True)
    def get(self):
        datasets = (
            g.session.query(Dataset, func.count(Annotation.id), func.count(Image.id))
            .outerjoin(Annotation)
            .outerjoin(Image)
            .group_by(Dataset.id)
            .all()
        )

        result = [
            {
                "name": ds.name,
                "keypoints": literal_eval(ds.keypoints),
                "created": ds.created_at,
                "images": img_cnt,
                "annotations": ann_cnt,
            }
            for ds, ann_cnt, img_cnt in datasets
        ]

        return result

    @api.expect(dataset_new)
    @api.marshal_with(message_model)
    @token_required
    def put(self, current_user: User):
        payload = request.json
        name = payload["datasetName"]
        keypoints = payload["keypoints"]
        animal_name = payload["animalName"]
        new_dataset = Dataset(
            dataset_name=name,
            keypoints=str(keypoints),
            animal_name=animal_name,
            user_id=current_user.id,
        )
        g.session.add(new_dataset)
        g.session.commit()
        return success_msg("new dataset created")
