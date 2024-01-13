from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    func,
)
from sqlalchemy.orm import relationship
from app.common.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(320), index=True, nullable=False, unique=True)
    true_name = Column(String(70), nullable=False)
    department = Column(String(100))
    bio = Column(String(200))
    password = Column(String(150), nullable=False)
    priority = Column(Integer, nullable=False, default=0)
    videos = relationship("Video", backref="user", passive_deletes=True)
    models = relationship("Model", backref="user", passive_deletes=True)
    datasets = relationship("Dataset", backref="user", passive_deletes=True)

    def __init__(
        self, email: str, password: str, true_name: str, department: str, bio: str
    ) -> None:
        self.email = email
        self.password = password
        self.true_name = true_name
        self.department = department
        self.bio = bio

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
