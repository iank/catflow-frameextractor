import configparser
import os


class Config:
    def __init__(self):
        # Load the configuration file
        config = configparser.ConfigParser()
        config.read(os.getenv("CONFIG_PATH", "frameextractor.ini"))

        # Service configuration
        self.SECRET_TOKEN = config.get("FrameExtractor", "SECRET_TOKEN")
        self.MODEL_NAME = config.get("FrameExtractor", "MODEL_NAME")
        self.THRESHOLD = config.getfloat("FrameExtractor", "THRESHOLD")

        # S3 configuration
        self.S3_CONFIG = {
            "bucket_name": config.get("S3", "BUCKET_NAME"),
            "endpoint_url": config.get("S3", "ENDPOINT_URL"),
            "access_key_id": config.get("S3", "ACCESS_KEY_ID"),
            "secret_access_key": config.get("S3", "SECRET_ACCESS_KEY"),
        }

        # Vector DB
        self.PG_CONFIG = {
            "host": config.get("PG", "HOST"),
            "dbname": config.get("PG", "DBNAME"),
            "username": config.get("PG", "USERNAME"),
            "password": config.get("PG", "PASSWORD"),
            "port": config.get("PG", "PORT"),
        }

        self.VDB_THRESHOLD = config.getfloat("VDB", "THRESHOLD")

        # Label Studio configuration
        self.PROJECT_ID = config.get("LabelStudioAPI", "PROJECT_ID")
        self.LABELSTUDIO_BASE_URL = config.get("LabelStudioAPI", "BASE_URL")
        self.LABELSTUDIO_AUTH_TOKEN = config.get("LabelStudioAPI", "AUTH_TOKEN")
