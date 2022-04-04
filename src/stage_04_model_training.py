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
import json
from src.utils.common import read_yaml
import joblib
STAGE = "Stage 04 Model Training"
logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
class model_training:
    def score_giver(self,C, D):
        E = pd.merge(C, D, how='outer', on='j')
        E.drop(columns=['j'], inplace=True)
        classifier=joblib.load("models/random_forest.joblib")
        q = classifier.predict(E.values)
        return Counter(q)
    def building_training_data(self,df):
        A = df[df['label'] == 1]
        A.loc[df['label'] == 1, 'join'] = 'j'
        B = df[df['label'] == 0]
        B.loc[df['label'] == 0, 'join'] = 'j'
        trainset1 = pd.merge(A, B, how='outer', on='join')
        trainset2 = pd.merge(B, A, how='outer', on='join')
        trainset = pd.merge(trainset1, trainset2, how='outer')
        return trainset
    def main(self,params_path,config_path):
        try:
            logging.info("Started 4th Stage")
            config=read_yaml(config_path)
            config1 = read_yaml(params_path)
            model_training_dir = config["featured"]["feature_dir"]
            model_training_file = config["featured"]["feature_processed"]
            model_dir = os.path.join(model_training_dir, model_training_file)
            model_file_path = config["featured"]["feature_file"]
            model_file = os.path.join(model_dir, model_file_path)
            df = pd.read_csv(model_file)
            product_list = df['product'].unique()
            logging.info("The unique products are {}".format(product_list))
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
            logging.info("Completed Building Training Data")
            X = train.iloc[:, :-1].values
            y = train.iloc[:, -1].values
            estimators=config1["estimators"]["random_forest"]["params"]
            n_estimators=estimators["n_estimators"]
            n_jobs=estimators["n_jobs"]
            oob_score=estimators["oob_score"]
            random_state=estimators["random_state"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
            classifier = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, oob_score=oob_score, random_state=random_state)
            classifier.fit(X_train, y_train)
            params_file=config["report"]["params"]
            scores_file=config["report"]["scores"]
            with open(params_file,'w') as f:
                params={
                "n_estimators":n_estimators,
                "n_jobs":n_jobs,
                "oob_score":oob_score,
                "random_state":random_state
                }
                json.dump(params,f,indent=4)
            logging.info("Successfully saved parameters file in {}".format(params_file))
            with open(scores_file,'w') as f:
                scores={
                    "Training_Accuracy": accuracy_score(y_train, classifier.predict(X_train)),
                    "Test_Accuracy" : accuracy_score(y_test, classifier.predict(X_test))}
                json.dump(scores, f, indent=4)
            logging.info("Successfully saved scores file in  {}".format(scores_file))
            model=config["model"]["dir"]
            file=config["model"]["file_name"]
            file_path= os.path.join(model,file)
            joblib.dump(classifier,file_path)
            logging.info("Successfully saved model file in {}".format(file_path))
            product_list = df['product'].unique()
            df['win'] = 0
            df['lose'] = 0
            df['review_score'] = 0.0
            df.reset_index(inplace=True, drop=True)
            for product in product_list:
                data = df[df['product'] == product]
                for indx in data.index:
                    review = df.iloc[indx, 3:-3]
                    review['j'] = 'jn'
                    C = pd.DataFrame([review])
                    D = data[data.index != indx].iloc[:, 3:-3]
                    D['j'] = 'jn'
                    score = self.score_giver(C, D)
                    df.at[indx, 'win'] = 0 if score.get(1) is None else score.get(1)
                    df.at[indx, 'lose'] = 0 if score.get(0) is None else score.get(0)
                    df.at[indx, 'review_score'] = float(0 if score.get(1) is None else score.get(1)) / len(data) * 1.0

            df = df.sort_values(by=['product', 'review_score'], ascending=False)
            df["rank"] = 0
            h = []
            for product in product_list:
                data = df[df['product'] == product]
                data = data.sort_values(by=['review_score'], ascending=False)
                d = 0
                for indx in data.index:
                    d = d + 1
                    data.loc[indx, "rank"] = d
                h.append(data)
            data = pd.concat(h).reset_index(drop=True)
            model_training_dir = config["featured"]["feature_dir"]
            model_training_file = config["featured"]["feature_processed"]
            model_dir = os.path.join(model_training_dir, model_training_file)
            model_file_path = config["featured"]["rank_file"]
            rank_file = os.path.join(model_dir, model_file_path)
            data.to_csv(rank_file,index=False)
            logging.info("Successfully saved the rank file in {}".format(rank_file))

        except Exception as e:
            logging.exception(e)
            raise e





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--params", "-p", default="params.yaml")
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    pre=model_training()
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        pre.main(params_path=parsed_args.params,config_path=parsed_args.config)
        logging.info(f">>>>>> stage {STAGE} completed!<<<<<<\n")
    except Exception as e:
        raise e