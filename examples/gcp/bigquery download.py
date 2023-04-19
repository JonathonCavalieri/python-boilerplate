#################################
# Import Libraries
from ayx import Alteryx
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from datetime import datetime
from hashlib import sha1
import pandas as pd
import os


from time import perf_counter


#################################
parameters = Alteryx.read("#1")
parameters


#################################
dt_run = datetime.now()
save_directory = parameters["save_directory"][0]
file_format = parameters["format"][0]
use_compression = parameters["use_compression"][0]

key_file_location = parameters["key_file"][0]
project_id = parameters["project"][0]

dataset_id = parameters["data_set"][0]
table_id = parameters["table"][0]
use_table = parameters["use_table"][0]
query = parameters["query"][0]

bucket_name = parameters["bucket"][0]


#################################
prefix = (table_id + str(dt_run)).encode("utf-8")
prefix = sha1(prefix).hexdigest()


#################################
# Set-up Credentials & Project
credentials = service_account.Credentials.from_service_account_file(key_file_location)
bq_client = bigquery.Client(credentials=credentials, project=project_id)
gcs_client = storage.Client(credentials=credentials, project=project_id)


#################################
if not use_table:
    temp_table = "temp_" + prefix
    query_job = f"""CREATE TABLE `{dataset_id}.{temp_table}` 
    OPTIONS(expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)) 
    AS {query}"""
    table_id = temp_table

    # run Query and copy data from result to temp table
    bq_client.query(query_job).result()


#################################
destination_uri = f"gs://{bucket_name}/{prefix}/{table_id}_*.{file_format}"
dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
table_ref = dataset_ref.table(table_id)
bucket = gcs_client.bucket(bucket_name)


#################################
destination_uri


#################################
job_config = bigquery.job.ExtractJobConfig()
job_config.destination_format = bigquery.DestinationFormat.AVRO

if use_compression:
    job_config.compression = bigquery.Compression.DEFLATE


#################################
start_time = perf_counter()
extract_job = bq_client.extract_table(
    table_ref, destination_uri, job_config=job_config, location="US"
)
extract_job.result()
end_time = perf_counter()
print(end_time - start_time)


#################################
files = []
print(save_directory)
if os.path.exists(save_directory):
    directory = save_directory + "file_download\\"
    os.makedirs(directory, exist_ok=True)

    start_time = perf_counter()
    for blob in gcs_client.list_blobs(bucket_name, prefix=prefix + "/"):
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        file_path = directory + file_split[-1]
        blob.download_to_filename(file_path)
        files.append(file_path)
    end_time = perf_counter()
    print(end_time - start_time)
else:
    raise ("Alteryx temporary folder not found")


#################################
df_out = pd.DataFrame({"File": files})
Alteryx.write(df_out, 1)


#################################
blobs = list(bucket.list_blobs(prefix=prefix + "/"))
bucket.delete_blobs(blobs)


#################################
if not use_table:
    temp_table_ref = dataset_ref.table(temp_table)
    bq_client.delete_table(temp_table_ref)
