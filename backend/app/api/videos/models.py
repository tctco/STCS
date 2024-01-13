from app.common.database import Base
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    func,
    Float,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from app.algorithm.models import Datum, TrackletStat


class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), index=True, nullable=False)  # token.ext
    original_name = Column(String(50), nullable=False)
    analyzed = Column(Boolean, default=False, nullable=False)
    frame_cnt = Column(Integer, nullable=False)
    fps = Column(Float, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    size = Column(Integer, nullable=False)  # bytes
    time_uploaded = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"))
    data = relationship(
        Datum.__name__, backref="video", passive_deletes=True, cascade="all, delete"
    )
    tracklet_stats = relationship(
        TrackletStat.__name__,
        backref="video",
        passive_deletes=True,
        cascade="all, delete",
    )

    def __init__(
        self,
        video_name: str,
        user_id: int,
        original_name: str,
        frame_cnt,
        fps,
        width,
        height,
        size,
    ) -> None:
        self.name = video_name
        self.user_id = user_id
        self.original_name = original_name
        self.frame_cnt = frame_cnt
        self.fps = fps
        self.width = width
        self.height = height
        self.size = size

    def __repr__(self) -> str:
        return f"<Video(id={self.id}, video_name={self.video_name}, user_id={self.user_id})>"
