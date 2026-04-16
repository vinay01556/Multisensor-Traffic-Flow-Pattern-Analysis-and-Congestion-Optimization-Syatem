import os
from dotenv import load_dotenv

# Inherit from environment defaults
load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "default-dev-key")
    CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")
    PORT = int(os.getenv("PORT", 5000))
    HOST = os.getenv("HOST", "0.0.0.0")

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
