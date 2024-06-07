from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    func,
)
from typing import Union
from app.common.database import Base


class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), index=True, nullable=False)
    dataset_id = Column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    category = Column(
        String(50), nullable=False
    )  # classification, detection, segmentation, etc.
    method = Column(
        String(50), nullable=True
    )  # algorithm RTMDet, MobileNetV2+SimpleBaseline, etc.
    params = Column(Text, nullable=True)
    trained = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    def __init__(
        self,
        model_name: str,
        dataset_id: int,
        user_id: int,
        category: str,
        method: str,
        trained: bool,
        params: Union[str, None],
    ) -> None:
        self.name = model_name
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.params = params
        self.category = category
        self.method = method
        self.trained = trained

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, model_name={self.model_name}, dataset_id={self.dataset_id}, user_id={self.user_id}, params={self.params})>"
