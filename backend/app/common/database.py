from app.config import DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker, declarative_base
import os

Base = declarative_base()

# Set up the database engine
engine = create_engine(DATABASE_URL, echo=False)

# Create a scoped session
Session = scoped_session(sessionmaker(bind=engine))


def init_db():
    from app.algorithm.models import Datum, TrackletStat
    from app.api.videos.models import Video
    from app.api.users.models import User
    from app.api.datasets.models import Dataset, Annotation, Image
    from app.api.models.models import Model

    if os.environ.get("DROP_ALL_DATA", "False") == "True":
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine, checkfirst=True)
    # if not engine.dialect.has_table(engine, "users"):
    #     print("Creating tables...")
    # else:
    #     print("Tables already exist.")
