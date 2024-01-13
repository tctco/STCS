from sqlalchemy import (
    Column,
    Integer,
    Text,
    Float,
    ForeignKey,
)
from app.common.database import Base


class Datum(Base):
    __tablename__ = "data"
    id = Column(Integer, primary_key=True)
    video_id = Column(
        Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame = Column(Integer, nullable=False, index=True)
    keypoints = Column(Text)
    keypoint_scores = Column(Text)
    bbox = Column(Text)  # xyxy
    bbox_score = Column(Float)
    raw_track_id = Column(Integer)
    track_id = Column(Integer)
    mask_area = Column(Integer)
    centroid = Column(Text)
    rle = Column(Text)

    def __init__(
        self,
        video_id: int,
        frame: int,
        keypoints: str,
        keypoint_scores: str,
        bbox: str,
        bbox_score: float,
        raw_track_id: int,
        track_id: int,
        mask_area: int,
        centroid: str,
        rle: str,
    ) -> None:
        self.video_id = video_id
        self.frame = frame
        self.keypoints = keypoints
        self.keypoint_scores = keypoint_scores
        self.bbox = bbox
        self.bbox_score = bbox_score
        self.raw_track_id = raw_track_id
        self.track_id = track_id
        self.mask_area = mask_area
        self.centroid = centroid
        self.rle = rle

    def __repr__(self) -> str:
        return f"<Datum(id={self.id}, video_id={self.video_id}, frame={self.frame}, raw_track_id={self.raw_track_id}, track_id={self.track_id}, mask_area={self.mask_area}, centroid: {self.centroid})>"


class TrackletStat(Base):
    __tablename__ = "tracklet_stats"
    id = Column(Integer, primary_key=True)
    video_id = Column(
        Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    track_id = Column(Integer, index=True, nullable=False)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    lifespan = Column(Integer, nullable=False)
    intervals = Column(Text, nullable=False)
    mask_area = Column(Integer, nullable=False)
    conf = Column(Float, nullable=False)
    distance = Column(Float, nullable=False)

    def __init__(
        self,
        video_id: int,
        track_id: int,
        start_frame: int,
        end_frame: int,
        lifespan: int,
        intervals: str,
        mask_area: int,
        conf: float,
        distance: float,
    ) -> None:
        self.video_id = video_id
        self.track_id = track_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.lifespan = lifespan
        self.intervals = intervals
        self.mask_area = mask_area
        self.conf = conf
        self.distance = distance

    def __repr__(self) -> str:
        return f"<TrackletStat(id={self.id}, video_id={self.video_id}, track_id={self.track_id}, start_frame={self.start_frame}, end_frame={self.end_frame}, lifespan={self.lifespan}, intervals={self.intervals}, mask_area={self.mask_area}, conf={self.conf})>"
