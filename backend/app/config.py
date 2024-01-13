import os

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data.db")
SECRET_KEY = os.environ.get("SECRET_KEY", "secret key")
