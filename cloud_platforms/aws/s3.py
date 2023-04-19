def cleanBucket(client, bucket, account_id, delete_folders=False):
    response = client.list_objects_v2(Bucket=bucket)

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
            client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": keys, "Quiet": True},
                ExpectedBucketOwner=account_id,
            )
    else:
        print(f"{bucket} already empty")
