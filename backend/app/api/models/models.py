from sqlalchemy import Column, ForeignKey, Integer, String
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

    def __init__(self, model_name: str, dataset_id: int, user_id: int) -> None:
        self.model_name = model_name
        self.dataset_id = dataset_id
        self.user_id = user_id

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, model_name={self.model_name}, dataset_id={self.dataset_id}, user_id={self.user_id})>"
