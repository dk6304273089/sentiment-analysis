import pandas as pd
import numpy as np
from joblib import load, dump
from copy import deepcopy
from statistics import mean
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import logging
import argparse
from src.utils.common import read_yaml

STAGE = "Stage 04 Model Training"
logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
class model_training:
    def building_training_data(self,df):
        A = df[df['label'] == 1]
        A.loc[df['label'] == 1, 'join'] = 'j'
        B = df[df['label'] == 0]
        B.loc[df['label'] == 0, 'join'] = 'j'
        trainset1 = pd.merge(A, B, how='outer', on='join')
        trainset2 = pd.merge(B, A, how='outer', on='join')
        trainset = pd.merge(trainset1, trainset2, how='outer')
        return trainset
    def main(self,config_path):
        config = read_yaml(config_path)
        model_training_dir = config["featured"]["feature_dir"]
        model_training_file = config["featured"]["feature_processed"]
        model_dir = os.path.join(model_training_dir, model_training_file)
        model_file_path = config["featured"]["feature_file"]
        model_file = os.path.join(model_dir, model_file_path)
        df = pd.read_csv(model_file)
        product_list = df['product'].unique()
        data_stack = []
        for product in product_list:
            temp = deepcopy(df[df['product'] == product].iloc[:, 2:])
            build_data = self.building_training_data(temp)
            print(product, len(temp), len(build_data))
            build_data.drop(columns=['join', 'label_y'], inplace=True)
            data = build_data.iloc[:, 1:]
            data['target'] = build_data.iloc[:, 0]
            data_stack.append(data)
        train = pd.concat(data_stack).reset_index(drop=True)
        X = train.iloc[:, :-1].values
        y = train.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
        print("Test Len:", len(X_test), " ", len(y_test))
        classifier = RandomForestClassifier(n_estimators=50, n_jobs=-1, oob_score=True, random_state=42)
        classifier.fit(X_train, y_train)

        print("Training Accuracy\n", accuracy_score(y_train, classifier.predict(X_train)))
        print("Test Accuracy\n", accuracy_score(y_test, classifier.predict(X_test)))





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    pre=model_training()
    try:
        pre.main(config_path=parsed_args.config)
    except Exception as e:
        raise e