import os
import yaml
import logging
import time
import pandas as pd
import json
import argparse
import logging
STAGE = "stage 01 Extracting Data"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    bucketname = config["bucketname"]
    filename = config["filename"]

    local_data_dir = config["source_download_dir"]["data_dir"]
    create_directories([local_data_dir])

    data_filename = config["source_download_dir"]["data_file"]
    local_data_filepath = os.path.join(local_data_dir, data_filename)

    print(local_data_filepath)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    main(config_path=parsed_args.config)
