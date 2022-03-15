import os
import yaml
import logging
import time
import pandas as pd
import io
import argparse
import logging
import sys
import boto3
from src.utils.common import read_yaml,create_directories

STAGE = "Stage 01 Extracting Data"
logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
def main(config_path,creditionals_path):
    ## read config files
    config = read_yaml(config_path)
    credentials=read_yaml(creditionals_path)
    #Getting aws_access_key
    secret_key=credentials["user"]["aws_access_key_id"]
    #Getting aws_secret_access_key
    aws_secret_access_key=credentials["user"]["aws_secret_access_key"]
    #Getting Bucket Name in S3
    bucketname = config["bucketname"]
    filename = config["filename"]

    local_data_dir = config["source_download_dir"]["data_dir"]
    create_directories([local_data_dir])
    #creating directory in data_filename_location
    data_filename = config["source_download_dir"]["data_file"]
    #saving the extracted data in local_data_filepath
    local_data_filepath = os.path.join(local_data_dir, data_filename)
    if sys.version_info[0] < 3:
        from StringIO import StringIO  # Python 2.x
    else:
        from io import StringIO
    try:
        logging.info("verifying the credentials")
        client = boto3.client('s3', aws_access_key_id=secret_key,
                              aws_secret_access_key=aws_secret_access_key)
        csv_obj = client.get_object(Bucket=bucketname, Key=filename)
        body = csv_obj['Body']
        logging.info("Started downloading the data")
        csv_string = body.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_string))
        df.to_csv(local_data_filepath,index=False)
        logging.info("Successfully saved the data in {}".format(local_data_filepath))
    except Exception as e:
        logging.exception(e)
        raise e

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--credentials", "-cr", default="configs/credentials.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config,creditionals_path=parsed_args.credentials)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
