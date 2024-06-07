from pathlib import Path
import os
from collections import defaultdict
import zipfile
import time
import shutil
from PIL import Image as PILImage
from flask import request, g, current_app
from app.common.decorators import token_required
from app.api.users.models import User
from app.common.messages import error_msg, success_msg
import numpy as np
from sqlalchemy import func
from ast import literal_eval
from flask_restx import Resource, Namespace, fields
import cv2
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from app.common.decorators import token_required
from app.api.shared_models import empty_input_model, message_model
from app.config import RESOURCE_PATH_URL
from .models import Dataset, Annotation, Image

# resource_path = os.environ.get("RESOURCE_PATH", "./static")
# IMAGE_ROOT_PATH = Path(f"{resource_path}/datasets")
# IMAGE_ROOT_PATH = current_app.config["IMAGE_ROOT_PATH"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def detect_mask_type(segmentation):
    if isinstance(segmentation, dict):
        if "counts" in segmentation and "size" in segmentation:
            return "rle"
    elif isinstance(segmentation, list):
        if all(isinstance(poly, list) for poly in segmentation):
            return "polygon"
    return "unknown"


def rle_to_polygon(rle):
    binary_mask = cocomask.decode(rle)
    polygons = cocomask.frPyObjects(
        binary_mask.astype(np.uint8), binary_mask.shape[0], binary_mask.shape[1]
    )
    polygon_points = []
    for polygon in polygons:
        points = polygon["all_points"]
        polygon_points.append(points)

    return polygon_points


class NullableInteger(fields.Raw):
    __schema_type__ = ["integer", "null"]
    __schema_example__ = "integer or null"

    def format(self, value):
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            raise fields.MarshallingError("Field should be an integer or null.")


def allowed_extension(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


api = Namespace("Datasets", description="Datasets")


dataset_info = api.model(
    "DatasetInfo",
    {
        "id": fields.Integer(required=True, description="Dataset id"),
        "name": fields.String(required=True, description="Dataset name"),
        "keypoints": fields.List(fields.String, required=True, description="Keypoints"),
        "animalName": fields.String(required=True, description="Animal name"),
        "created": fields.DateTime(required=True, description="Created time"),
        "images": fields.Integer(required=True, description="Number of images"),
        "annotations": fields.Integer(
            required=True, description="Number of annotations"
        ),
    },
)

new_dataset_model = api.model(
    "DatasetNew",
    {
        "name": fields.String(required=True, description="Dataset name"),
        "keypoints": fields.List(fields.String, required=True, description="Keypoints"),
        "animalName": fields.String(required=True, description="Animal name"),
    },
)

annotation_model = api.model(
    "Annotation",
    {
        "id": fields.Integer(required=True, description="Annotation id"),
        "imageId": fields.Integer(required=True, description="Image id"),
        "keypoints": fields.List(fields.Float, required=True, description="Keypoints"),
        "area": fields.Float(required=True, description="Area"),
        "bbox": fields.List(
            fields.Float,
            required=True,
            description="Bounding box, should be in xyxy format",
        ),
        "polygon": fields.List(
            fields.List(fields.Float),
            required=True,
            description="polygon segmentation",
        ),
    },
)

image_model = api.model(
    "Image",
    {
        "id": fields.Integer(required=True, description="Image id"),
        "url": fields.String(required=True, description="Image url"),
        "difficult": fields.Boolean(
            required=True,
            description="If the image is considered difficult for a deep learning model",
        ),
        "annotations": fields.List(
            fields.Nested(annotation_model), required=True, description="Annotations"
        ),
    },
)


new_annotation_model = api.model(
    "NewAnnotation",
    {
        "id": fields.Integer(
            required=True, description="Annotation id. If null, create new annotation"
        ),
        "keypoints": fields.List(fields.Float, required=True, description="Keypoints"),
        "polygon": fields.List(
            fields.List(fields.Float),
            required=True,
            description="polygon segmentation",
        ),
    },
)
update_image_model = api.model(
    "NewImage", {"annotations": fields.List(fields.Nested(new_annotation_model))}
)


@api.route("/")
class DatasetListResource(Resource):
    @api.expect(empty_input_model, validate=False)
    @api.marshal_with(dataset_info, as_list=True)
    @token_required
    def get(self, current_user: User):
        subquery_annotations = (
            g.session.query(
                Annotation.dataset_id,
                func.count(Annotation.id).label("annotations_count"),
            )
            .group_by(Annotation.dataset_id)
            .subquery()
        )
        subquery_images = (
            g.session.query(
                Image.dataset_id, func.count(Image.id).label("images_count")
            )
            .group_by(Image.dataset_id)
            .subquery()
        )
        datasets = (
            g.session.query(
                Dataset,
                func.coalesce(subquery_annotations.c.annotations_count, 0),
                func.coalesce(subquery_images.c.images_count, 0),
            )
            .outerjoin(
                subquery_annotations, subquery_annotations.c.dataset_id == Dataset.id
            )
            .outerjoin(subquery_images, subquery_images.c.dataset_id == Dataset.id)
            .filter(Dataset.user_id == current_user.id)
            .all()
        )

        result = [
            {
                "id": ds.id,
                "name": ds.name,
                "keypoints": literal_eval(ds.keypoints),
                "animalName": ds.animal_name,
                "created": ds.created_at,
                "images": img_cnt,
                "annotations": ann_cnt,
            }
            for ds, ann_cnt, img_cnt in datasets
        ]

        return result

    @api.expect(new_dataset_model)
    @api.marshal_with(dataset_info)
    @token_required
    def post(self, current_user: User):
        """create a new dataset

        Args:
            current_user (User): current user

        Returns:
            message_model: success message
        """
        payload = request.json
        name = payload["name"]
        keypoints = payload["keypoints"]
        animal_name = payload["animalName"]
        new_dataset = Dataset(
            dataset_name=name,
            keypoints=str(keypoints).replace(" ", ""),
            animal_name=animal_name,
            user_id=current_user.id,
        )
        g.session.add(new_dataset)
        g.session.commit()
        IMAGE_ROOT_PATH = Path(current_app.config["IMAGE_ROOT_PATH"])
        (IMAGE_ROOT_PATH / str(current_user.id) / str(new_dataset.id)).mkdir(
            parents=True, exist_ok=True
        )
        return {
            "id": new_dataset.id,
            "name": new_dataset.name,
            "keypoints": literal_eval(new_dataset.keypoints),
            "animalName": new_dataset.animal_name,
            "created": new_dataset.created_at,
            "images": 0,
            "annotations": 0,
        }


@api.route("/<int:dataset_id>")
class DatasetResource(Resource):
    @api.marshal_with(dataset_info)
    @token_required
    def get(self, current_user: User, dataset_id: int):
        dataset: Dataset = (
            g.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        )
        if dataset is None:
            return error_msg("dataset not found"), 400
        if dataset.user_id != current_user.id:
            return error_msg("permission denied"), 400
        ann_cnt = (
            g.session.query(func.count(Annotation.id))
            .filter(Annotation.dataset_id == dataset_id)
            .scalar()
        )
        img_cnt = (
            g.session.query(func.count(Image.id))
            .filter(Image.dataset_id == dataset_id)
            .scalar()
        )
        result = {
            "id": dataset.id,
            "name": dataset.name,
            "keypoints": literal_eval(dataset.keypoints),
            "animalName": dataset.animal_name,
            "created": dataset.created_at,
            "images": img_cnt,
            "annotations": ann_cnt,
        }
        return result

    @api.marshal_with(message_model)
    @token_required
    def delete(self, current_user: User, dataset_id: int):
        dataset = g.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            return error_msg("dataset not found"), 400
        if dataset.user_id != current_user.id:
            return error_msg("permission denied"), 400
        g.session.delete(dataset)
        g.session.commit()
        IMAGE_ROOT_PATH = current_app.config["IMAGE_ROOT_PATH"]
        if Path(IMAGE_ROOT_PATH / str(current_user.id) / str(dataset_id)).exists():
            shutil.rmtree(IMAGE_ROOT_PATH / str(current_user.id) / str(dataset_id))
        return success_msg("dataset deleted")

    @api.expect(new_dataset_model)
    @api.marshal_with(message_model)
    @token_required
    def post(self, current_user: User, dataset_id: int):
        dataset = g.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            return error_msg("dataset not found"), 400
        if dataset.user_id != current_user.id:
            return error_msg("permission denied"), 400
        payload = request.json
        dataset.dataset_name = payload["name"]
        dataset.keypoints = str(payload["keypoints"])
        dataset.animal_name = payload["animalName"]
        g.session.commit()
        return success_msg("dataset updated")


@api.route("/<int:dataset_id>/images/<int:image_id>")
class ImageResource(Resource):
    @api.marshal_with(image_model)
    @token_required
    def get(self, current_user: User, dataset_id: int, image_id: int):
        dataset = (
            g.session.query(Dataset)
            .filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
            .first()
        )
        if dataset is None:
            return error_msg("dataset not found"), 404
        image = (
            g.session.query(Image)
            .filter(Image.id == image_id, Image.dataset_id == dataset_id)
            .first()
        )
        if image is None:
            return error_msg("image not found"), 400
        if image.dataset.user_id != current_user.id:
            return error_msg("permission denied"), 400
        return {
            "id": image.id,
            "url": f"{RESOURCE_PATH_URL}/images/{current_user.id}/{image.dataset_id}/{image.name}",
            "difficult": image.difficult,
            "annotations": [
                {
                    "id": ann.id,
                    "image_id": ann.image_id,
                    "keypoints": literal_eval(ann.keypoints) if ann.keypoints else [],
                    "area": ann.area,
                    "bbox": literal_eval(ann.bbox) if ann.bbox else [],
                    "polygon": literal_eval(ann.polygon) if ann.bbox else [],
                }
                for ann in image.annotations
            ],
        }

    @api.marshal_with(message_model)
    @token_required
    def delete(self, current_user: User, dataset_id: int, image_id: int):
        dataset = (
            g.session.query(Dataset)
            .filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
            .first()
        )
        if dataset is None:
            return error_msg("dataset not found"), 404
        image = (
            g.session.query(Image)
            .filter(Image.id == image_id, Image.dataset_id == dataset_id)
            .first()
        )
        if image is None:
            return error_msg("image not found"), 400
        if image.dataset.user_id != current_user.id:
            return error_msg("permission denied"), 400
        g.session.delete(image)
        g.session.commit()
        IMAGE_ROOT_PATH = current_app.config["IMAGE_ROOT_PATH"]
        if Path(
            IMAGE_ROOT_PATH
            / str(current_user.id)
            / str(image.dataset_id)
            / str(image.name)
        ).exists():
            shutil.rmtree(IMAGE_ROOT_PATH / str(current_user.id) / str(image_id))
        return success_msg("image deleted")

    def _get_bbox_from_polygon(self, poly) -> list:
        x = [poly[i] for i in range(0, len(poly), 2)]
        y = [poly[i] for i in range(1, len(poly), 2)]
        return [min(x), min(y), max(x), max(y)]

    def _collect_xy_from_polygon_keypoints(self, ann):
        all_xy = []
        if ann["polygon"] is not None:
            for poly in ann["polygon"]:
                all_xy.extend(poly)
        if ann["keypoints"] is not None:
            for i in range(0, len(ann["keypoints"]), 3):
                if ann["keypoints"][i + 2] == 0:
                    continue
                all_xy.extend(ann["keypoints"][i : i + 2])
        return all_xy

    @api.marshal_with(message_model)
    @api.expect(update_image_model)
    @token_required
    def post(self, current_user: User, dataset_id: int, image_id: int):
        dataset = (
            g.session.query(Dataset)
            .filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
            .first()
        )
        if dataset is None:
            return error_msg("dataset not found"), 404
        payload = request.json
        image = (
            g.session.query(Image)
            .filter(Image.id == image_id, Image.dataset_id == dataset_id)
            .first()
        )
        if image is None:
            return error_msg("image not found"), 400
        if image.dataset.user_id != current_user.id:
            return error_msg("permission denied"), 400
        # new_ann_cnt = 0
        modified_ann_cnt = 0
        ann_not_found = 0

        for ann in payload["annotations"]:
            current_app.logger.debug(f"ann {ann}")
            all_xy = self._collect_xy_from_polygon_keypoints(ann)
            if len(all_xy) > 0:
                ann["bbox"] = self._get_bbox_from_polygon(all_xy)
            else:
                ann["bbox"] = None

            if ann["polygon"] is not None:
                area = sum(
                    cv2.contourArea(np.array(poly, dtype=np.float32).reshape((-1, 2)))
                    for poly in ann["polygon"]
                )  # TODO: multiple polygons
            elif ann["bbox"] is not None:
                area = (ann["bbox"][2] - ann["bbox"][0]) * (
                    ann["bbox"][3] - ann["bbox"][1]
                )
            else:
                area = None
            keypoints = None if ann["keypoints"] is None else str(ann["keypoints"])
            bbox = None if ann["bbox"] is None else str(ann["bbox"])
            poly = None if ann["polygon"] is None else str(ann["polygon"])
            annotation = (
                g.session.query(Annotation).filter(Annotation.id == ann["id"]).first()
            )
            if annotation is None:
                ann_not_found += 1
                continue
            annotation.keypoints = keypoints
            annotation.area = area
            annotation.bbox = bbox
            annotation.polygon = poly
            modified_ann_cnt += 1
        g.session.commit()
        if modified_ann_cnt == 0:
            return error_msg("No valid annotation"), 400
        return success_msg(
            f"Updated {modified_ann_cnt} annotations, {ann_not_found} annotations not found"
        )

    @token_required
    @api.marshal_with(annotation_model)
    def put(self, current_user: User, dataset_id: int, image_id: int):
        dataset = (
            g.session.query(Dataset)
            .filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
            .first()
        )
        if dataset is None:
            return error_msg("dataset not found"), 404
        image = (
            g.session.query(Image)
            .filter(
                Image.id == image_id,
                Image.dataset_id == dataset_id,
                Image.dataset_id == dataset_id,
            )
            .first()
        )
        if image is None:
            return error_msg("image not found"), 404
        new_annotation = Annotation(image_id, None, None, None, None, dataset_id)
        g.session.add(new_annotation)
        g.session.commit()
        return {
            "id": new_annotation.id,
            "imageId": new_annotation.image_id,
            "keypoints": [],
            "area": None,
            "bbox": [],
            "polygon": [],
        }


@api.route("/<int:dataset_id>/coco")
class CocoDataset(Resource):
    @token_required
    def post(self, current_user: User, dataset_id: int):
        """upload coco file"""
        # check dataset validity
        dataset = g.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            return error_msg("dataset not found"), 400
        if dataset.user_id != current_user.id:
            return error_msg("permission denied"), 400
        upload_file = request.files.get("file", None)
        if upload_file is None:
            return error_msg("No file part"), 400
        if not upload_file.filename.endswith(".zip"):
            return error_msg("Invalid file type. File has to be .zip"), 400
        IMAGE_ROOT_PATH = current_app.config["IMAGE_ROOT_PATH"]
        zip_path = os.path.join(IMAGE_ROOT_PATH, "temp.zip")
        upload_file.save(zip_path)
        try:
            img_cnt, ann_cnt = self._process_zip(zip_path, current_user.id, dataset_id)
        except Exception as e:
            return error_msg(str(e)), 400
        return success_msg(f"Saved {img_cnt} images, {ann_cnt} annotations")

    def _check_annotations(self, coco: COCO, temp_dir: Path, num_keypoints: int):
        result = defaultdict(list)
        annotations = coco.loadAnns(coco.getAnnIds())
        for ann in annotations:
            if (
                "keypoints" in ann
                and ann["keypoints"]
                and "segmentation" in ann
                and ann["segmentation"]
                and len(ann["keypoints"]) == num_keypoints * 3
            ):
                img_name = coco.loadImgs(ann["image_id"])[0]["file_name"]
                img_path = temp_dir / img_name
                if not os.path.exists(img_path):
                    continue
                result[img_path].append(ann)
        return result

    def _process_zip(self, zip_path, user_id: int, dataset_id: int) -> int:
        """process uploaded coco file
        TODO: maybe merge /coco with /images

        Args:
            zip_path (str): uploaded tmp zip path
            user_id (int): uploaded user id
            dataset_id (int): dataset id

        Returns:
            int: len valid images
        """
        IMAGE_ROOT_PATH = Path(current_app.config["IMAGE_ROOT_PATH"])
        temp_dir = IMAGE_ROOT_PATH / "temp" / str(time.time_ns())
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(str(temp_dir))

            annotations_path = temp_dir / "data.json"
            coco = COCO(annotations_path)

            # check if keypoints are the same with db.dataset keypoints
            dataset = g.session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset is None:
                raise Exception("dataset not found")
            num_keypoints = len(literal_eval(dataset.keypoints))
            valid_images = self._check_annotations(coco, temp_dir, num_keypoints)

            # copy and create image in database
            img_cnt, ann_cnt = 0, 0
            for image_path, anns in valid_images.items():
                ext = image_path.suffix.lower()
                with PILImage.open(image_path) as img:
                    width, height = img.size
                new_image_item = Image(
                    str(time.time_ns()), width, height, False, dataset_id
                )
                g.session.add(new_image_item)
                g.session.commit()
                image_name = f"{new_image_item.id}{ext}"
                new_image_item.name = image_name
                g.session.commit()
                shutil.copy(
                    image_path,
                    os.path.join(
                        IMAGE_ROOT_PATH, f"{user_id}/{dataset_id}/{image_name}"
                    ),
                )
                img_cnt += 1
                for ann in anns:
                    keypoints = str(ann["keypoints"]).replace(" ", "")
                    mask_type = detect_mask_type(ann["segmentation"])
                    if mask_type == "rle":
                        segm = rle_to_polygon(ann["segmentation"])
                        segm = str(segm).replace(" ", "")
                    elif mask_type == "polygon":
                        segm = ann["segmentation"]
                        segm = str(segm).replace(" ", "")
                    else:
                        segm = None
                    bbox = ann.get("bbox", None)
                    if bbox is not None:
                        bbox = str(bbox).replace(" ", "")
                    new_annotation = Annotation(
                        new_image_item.id,
                        keypoints,
                        ann.get("area", None),
                        bbox,
                        segm,
                        dataset_id,
                    )
                    g.session.add(new_annotation)
                    ann_cnt += 1
                g.session.commit()

            shutil.rmtree(temp_dir)
            os.remove(zip_path)
            return img_cnt, ann_cnt

        except Exception as e:
            shutil.rmtree(temp_dir)
            os.remove(zip_path)
            raise e


@api.route("/<int:dataset_id>/images/<int:image_id>/<int:annotation_id>")
class DatasetImageAnnotations(Resource):
    @token_required
    @api.marshal_with(annotation_model)
    def delete(
        self, current_user: User, dataset_id: int, image_id: int, annotation_id: int
    ):
        annotation = (
            g.session.query(Annotation)
            .filter(Annotation.id == annotation_id, Annotation.image_id == image_id)
            .first()
        )
        if annotation is None:
            return error_msg("annotation not found"), 400
        if annotation.image.dataset.user_id != current_user.id:
            return error_msg("permission denied"), 400
        g.session.delete(annotation)
        g.session.commit()
        return success_msg("annotation deleted")


@api.route("/<int:dataset_id>/images")
class DatasetImagesResource(Resource):
    """add or get images in a dataset"""

    @api.marshal_with(image_model, as_list=True)
    @token_required
    def get(self, current_user: User, dataset_id: int):
        images = g.session.query(Image).filter(Image.dataset_id == dataset_id).all()
        if images is None:
            return error_msg("images not found"), 400
        if len(images) > 0 and images[0].dataset.user_id != current_user.id:
            return error_msg("permission denied"), 400
        return [
            {
                "id": image.id,
                "url": f"{RESOURCE_PATH_URL}/images/{current_user.id}/{dataset_id}/{image.name}",
                "difficult": image.difficult,
                "annotations": [
                    {
                        "id": ann.id,
                        "image_id": ann.image_id,
                        "keypoints": (
                            literal_eval(ann.keypoints) if ann.keypoints else []
                        ),
                        "area": ann.area,
                        "bbox": literal_eval(ann.bbox) if ann.bbox else [],
                        "polygon": literal_eval(ann.polygon) if ann.bbox else [],
                    }
                    for ann in image.annotations
                ],
            }
            for image in images
        ]

    @api.marshal_with(message_model)
    @token_required
    def post(self, current_user: User, dataset_id: int):
        dataset = g.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            return error_msg("dataset not found"), 400
        if "file" not in request.files:
            return error_msg("No file part"), 400
        files = request.files.getlist("file")
        if not files or files[0].filename == "":
            return error_msg("No selected file"), 400

        IMAGE_ROOT_PATH = current_app.config["IMAGE_ROOT_PATH"]
        saved_cnt = 0
        save_path = IMAGE_ROOT_PATH / str(current_user.id) / str(dataset.id)
        save_path.mkdir(parents=True, exist_ok=True)
        for file in files:
            if file and allowed_extension(file.filename):
                ext = file.filename.rsplit(".", 1)[1].lower()
                new_image_item = Image(str(time.time_ns()), 0, 0, False, dataset.id)
                g.session.add(new_image_item)
                g.session.commit()
                new_image_item.name = f"{new_image_item.id}{ext}"
                file.save(save_path / new_image_item.name)
                with PILImage.open(save_path / new_image_item.name) as img:
                    new_image_item.width, new_image_item.height = img.size
                saved_cnt += 1
        g.session.commit()
        if saved_cnt == 0:
            return error_msg("No valid file"), 400
        return success_msg(f"Saved {saved_cnt} images")
