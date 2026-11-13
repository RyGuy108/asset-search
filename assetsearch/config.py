import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/assets")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
MODEL_DIR = os.environ.get("ASSET_ENCODER_DIR", os.path.join(os.path.dirname(__file__), "..", "models", "encoder"))
MAGIC_HOUR_API_KEY = os.environ.get("MAGIC_HOUR_API_KEY")
MEDIA_DIR = os.environ.get("MEDIA_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "media"))
