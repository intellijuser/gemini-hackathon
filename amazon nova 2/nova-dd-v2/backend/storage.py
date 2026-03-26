import os
from abc import ABC, abstractmethod
from typing import Optional

import boto3

from config import settings


class ObjectStorage(ABC):
    @abstractmethod
    def put_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_bytes(self, uri: str) -> bytes:
        raise NotImplementedError()

    @abstractmethod
    def delete(self, uri: str) -> None:
        raise NotImplementedError()


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _full_path(self, key: str) -> str:
        safe_key = key.replace("\\", "/").lstrip("/")
        full = os.path.join(self.root, safe_key)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        return full

    def put_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        full = self._full_path(key)
        with open(full, "wb") as f:
            f.write(content)
        return f"file://{os.path.abspath(full)}"

    def get_bytes(self, uri: str) -> bytes:
        if not uri.startswith("file://"):
            raise ValueError("Invalid local URI.")
        path = uri.replace("file://", "", 1)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, uri: str) -> None:
        if not uri.startswith("file://"):
            return
        path = uri.replace("file://", "", 1)
        if os.path.exists(path):
            os.remove(path)


class S3ObjectStorage(ObjectStorage):
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )

    def put_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
            ServerSideEncryption="AES256",
        )
        return f"s3://{self.bucket}/{key}"

    def get_bytes(self, uri: str) -> bytes:
        if not uri.startswith("s3://"):
            raise ValueError("Invalid s3 URI.")
        no_scheme = uri.replace("s3://", "", 1)
        bucket, key = no_scheme.split("/", 1)
        resp = self.client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    def delete(self, uri: str) -> None:
        if not uri.startswith("s3://"):
            return
        no_scheme = uri.replace("s3://", "", 1)
        bucket, key = no_scheme.split("/", 1)
        self.client.delete_object(Bucket=bucket, Key=key)


def build_object_storage() -> ObjectStorage:
    if settings.object_store_mode == "s3":
        return S3ObjectStorage(settings.object_store_bucket)
    return LocalObjectStorage(settings.local_storage_root)
