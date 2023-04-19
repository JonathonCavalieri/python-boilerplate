from boto3 import client
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
from pathlib import Path

messages = []
base_directory = r"xxxxx"
account_id = "xxxx"

# list all buckets
# This will download all buckets for account, may need to filter in future if want to use other storage
try:
    s3 = client("s3", aws_access_key_id="xxxx", aws_secret_access_key="xxx")
    paginator = s3.get_paginator("list_objects_v2")
    buckets_response = s3.list_buckets()
except Exception as err:
    messages.append(
        {
            "type": "pythonError",
            "message": err,
            "source": f"S3BucketDownload-listBuckets",
        }
    )


def cleanBucket(bucket, account_id, delete_folders=False):
    response = s3.list_objects_v2(Bucket=bucket)

    if "Contents" in response.keys():
        print(f"deleting contents of {bucket}")
        if delete_folders:
            keys = [
                {key: x[key] for key in x.keys() if key == "Key"}
                for x in response["Contents"]
            ]
        else:
            keys = [
                {
                    key: x[key]
                    for key in x.keys()
                    if key == "Key" and not x[key].endswith("/")
                }
                for x in response["Contents"]
            ]

        if keys == [{}]:
            print(f"{bucket} has no files to delete")
        else:
            s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": keys, "Quiet": True},
                ExpectedBucketOwner=account_id,
            )
    else:
        print(f"{bucket} already empty")


buckets = [x["Name"] for x in buckets_response["Buckets"]]

files = Path(base_directory).glob("**/*")
files = [x for x in files if x.is_file()]

for file in files:
    relative_path = file.relative_to(base_directory)
    bucket = relative_path.parts[0]
    key = "input/" + str(relative_path.relative_to(bucket)).replace("\\", "/")
    path = str(file)
    # print( str(file).replace(base_directory, ''))
    if bucket not in buckets:
        messages.append(
            {
                "type": "Error",
                "message": f"No bucket matched folder: {bucket}",
                "source": f"{path}",
            }
        )
        continue
    print(f"Uploading file {relative_path}")
    s3.upload_file(path, bucket, key)
