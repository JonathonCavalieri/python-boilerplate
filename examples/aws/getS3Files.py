from boto3 import client
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
from pathlib import Path

messages = []
base_directory = r"xxx"

# list all buckets
# This will download all buckets for account, may need to filter in future if want to use other storage
try:
    s3 = client("s3", aws_access_key_id="xxxx", aws_secret_access_key="xxxx")
    paginator = s3.get_paginator("list_objects_v2")
    buckets = s3.list_buckets()
except Exception as err:
    messages.append(
        {
            "type": "pythonError",
            "message": err,
            "source": f"S3BucketDownload-listBuckets",
        }
    )


# Delete Old files from the
try:
    files = Path(base_directory).glob("**/*")
    files = [x for x in files if x.is_file()]
    for file in files:
        file.unlink()
        messages.append(
            {
                "type": "info",
                "message": f"Deleted old file: {file}",
                "source": f"S3BucketDownload-deleteOldFiles",
            }
        )
    print(files)
except Exception as err:
    messages.append(
        {
            "type": "pythonError",
            "message": err,
            "source": f"S3BucketDownload-deleteOldFiles",
        }
    )

try:
    # Get a list of all the files and folders in all S3 buckets
    objects = []

    for bucket in buckets["Buckets"]:
        bucket_name = bucket["Name"]

        download = False
        # check to see if bucket has tag to download
        try:
            tags = s3.get_bucket_tagging(Bucket=bucket_name)
            for tag in tags["TagSet"]:
                if tag["Key"] == "Download" and tag["Value"] == "True":
                    download = True
        except ClientError:
            pass
        # skip bucket if it doesnt have tag
        if not download:
            continue

        messages.append(
            {
                "type": "info",
                "message": f"Getting files for bucket: {bucket_name}",
                "source": f"S3BucketDownload-listFiles",
            }
        )
        pages = paginator.paginate(Bucket=bucket_name)
        for page in pages:
            for obj in page["Contents"]:
                obj["Bucket"] = bucket_name
                objects.append(obj)
    # convert to dataframe
    objects = pd.DataFrame.from_dict(objects)
except Exception as err:
    messages.append(
        {"type": "pythonError", "message": err, "source": f"S3BucketDownload-listFiles"}
    )


try:
    # Add additional columns in and filter to only files
    objects = objects[objects["Size"] != 0].copy()
    objects["Client"] = np.where(
        objects["Bucket"].str.contains("fva-client-"),
        objects["Bucket"].str.replace("fva-client-", ""),
        np.nan,
    )
    objects["OutputLocation"] = (
        base_directory
        + np.where(
            objects["Client"].isnull(),
            "",
            "02-client-data\\" + objects["Client"] + "\\",
        )
        + objects["Key"].str.replace("/", "\\")
    )
    objects["Folder"] = objects["OutputLocation"].apply(lambda x: str(Path(x).parent))

    # Create any folders that dont exist
    folders = objects["Folder"].unique()
    for folder in folders:
        messages.append(
            {
                "type": "info",
                "message": f"Making directoy: {folder}",
                "source": f"S3BucketDownload-downloadFiles",
            }
        )
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Download files from s3 buckets
    for index, file in objects.iterrows():
        location = file["OutputLocation"]
        key = file["Key"]
        bucket = file["Bucket"]
        messages.append(
            {
                "type": "info",
                "message": f"Downloading file: {key}",
                "source": f"S3Bucket: {bucket}",
            }
        )
        s3.download_file(bucket, key, location)
except Exception as err:
    messages.append(
        {
            "type": "pythonError",
            "message": err,
            "source": f"S3BucketDownload-downloadFiles",
        }
    )
