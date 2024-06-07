from app.common.database import Base
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    Float,
    Boolean,
    func,
)
from sqlalchemy.orm import relationship
from typing import Union


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
    images = relationship(
        "Image",
        backref="dataset",
        passive_deletes=True,
        cascade="all, delete, delete-orphan",
    )
    models = relationship(
        "Model",
        backref="dataset",
        passive_deletes=True,
        cascade="all, delete, delete-orphan",
    )
    created_at = Column(DateTime, default=func.now(), nullable=False)

    def __init__(
        self,
        dataset_name: str,
        animal_name: int,
        keypoints: str,
        user_id: int,
    ) -> None:
        self.name = dataset_name
        self.animal_name = animal_name
        self.keypoints = keypoints
        self.user_id = user_id

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, dataset_name={self.name}, animal_name={self.animal_name}, keypoints={self.keypoints}, user_id={self.user_id})>"


class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True)
    image_id = Column(
        Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    keypoints = Column(Text)
    area = Column(Float)
    bbox = Column(Text)
    polygon = Column(Text)
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    create_at = Column(DateTime, default=func.now(), nullable=False)
    update_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    def __init__(
        self,
        image_id: int,
        keypoints: Union[str, None],
        area: Union[float, None],
        bbox: Union[str, None],
        polygon: Union[str, None],
        dataset_id: int,
    ) -> None:
        self.image_id = image_id
        self.keypoints = keypoints
        self.area = area
        self.bbox = bbox
        self.polygon = polygon
        self.dataset_id = dataset_id


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    name = Column(String(80), unique=True, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    difficult = Column(Boolean, default=False, nullable=False)
    annotations = relationship("Annotation", backref="image", lazy=True)
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"id: {self.id}, name: {self.name}, difficult: {self.difficult}"

    def __init__(
        self, name: str, width: int, height: int, difficult: bool, dataset_id: int
    ) -> None:
        self.name = name
        self.difficult = difficult
        self.dataset_id = dataset_id
        self.width = width
        self.height = height
