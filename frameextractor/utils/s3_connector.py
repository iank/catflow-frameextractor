import boto3
from flask import current_app

class S3Connector:
    def __init__(self, endpoint_url, access_key_id, secret_access_key):
        self.s3 = self._connect_s3(endpoint_url, access_key_id, secret_access_key)

    def _connect_s3(self, endpoint_url, access_key_id, secret_access_key):
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key
        )
        s3 = session.client('s3', endpoint_url=endpoint_url)
        return s3

    def upload_file(self, bucket_name, file_path, object_name):
        try:
            self.s3.upload_file(file_path, bucket_name, object_name)
            s3_uri = f"{bucket_name}/{object_name}"
            return s3_uri
        except Exception as e:
            current_app.logger.error("Error uploading file: ", e)
            return None
