import os

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data.db")
SECRET_KEY = os.environ.get("SECRET_KEY", "secret key")

RESOURCE_PATH = os.environ.get("RESOURCE_PATH", "static")
if RESOURCE_PATH == "static":
    RESOURCE_PATH_URL = "/static"
else:
    RESOURCE_PATH_URL = "/resources"
