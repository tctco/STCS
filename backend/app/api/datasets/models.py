from app.common.database import Base
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Float, func
from sqlalchemy.orm import relationship


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), index=True, nullable=False)
    animal_name = Column(String(50))
    keypoints = Column(Text, nullable=False)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    annotations = relationship("Annotation", backref="dataset", passive_deletes=True)
    images = relationship("Image", backref="dataset", passive_deletes=True)
    videos = relationship("Video", backref="dataset", passive_deletes=True)
    models = relationship("Model", backref="dataset", passive_deletes=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    def __init__(
        self,
        dataset_name: str,
        animal_name: int,
        keypoints: str,
        user_id: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.animal_name = animal_name
        self.keypoints = keypoints
        self.user_id = user_id

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, dataset_name={self.dataset_name}, animal_name={self.animal_name}, keypoints_num={self.keypoints_num}, keypoint_names={self.keypoint_names}, user_id={self.user_id})>"


class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    keypoints = Column(Text, nullable=False)
    area = Column(Float, nullable=False)
    bbox = Column(Text, nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    create_at = Column(DateTime, default=func.now(), nullable=False)

    def __init__(
        self, image_id: int, keypoints: str, area: float, bbox: str, dataset_id: int
    ) -> None:
        self.image_id = image_id
        self.keypoints = keypoints
        self.area = area
        self.bbox = bbox
        self.dataset_id = dataset_id


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    name = Column(String(80), unique=True, nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id"))
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    annotations = relationship("Annotation", backref="image", lazy=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)

    def __repr__(self) -> str:
        return f"id: {self.id}, name: {self.name}"

    def __init__(self, name: str, video_id: int, width: int, height: int) -> None:
        self.name = name
        self.video_id = video_id
        self.width = width
        self.height = height
