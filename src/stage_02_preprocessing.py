import os
import yaml
import logging
import time
import math
from langdetect import detect
import pandas as pd
import io
import pickle
import jellyfish
import spacy
nlp = spacy.load("en_core_web_sm")
import argparse
import logging
import sys
import boto3
from io import StringIO
from src.utils.common import read_yaml,create_directories
from copy import deepcopy
STAGE = "Stage 02 Processing Data"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
class preprocessing:
    def language_detection(self, text):
        '''
        :param text: Text for which to detect language
        :return: `hi` or `bi` or `en`, etc
        Source: https://github.com/Mimino666/langdetect
        '''
        return detect(text)

    def gibberish_detection(self, l):
        '''
        Input: String
        prefix_path: path of gibberish pickle weights
        Output: True or False
        '''
        model_data = pickle.load(open('models/gib_model.pki', 'rb'))
        accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
        pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

        def normalize(line):
            """ Return only the subset of chars from accepted_chars.
            This helps keep the  model relatively small by ignoring punctuation,
            infrequenty symbols, etc. """
            return [c.lower() for c in line if c.lower() in accepted_chars]

        def ngram(n, l):
            """ Return all n grams from l after normalizing """
            filtered = normalize(l)
            for start in range(0, len(filtered) - n + 1):
                yield ''.join(filtered[start:start + n])

        def avg_transition_prob(l, log_prob_mat):
            """ Return the average transition prob from l through log_prob_mat. """
            log_prob = 0.0
            transition_ct = 0
            for a, b in ngram(2, l):
                log_prob += log_prob_mat[pos[a]][pos[b]]
                transition_ct += 1
            # The exponentiation translates from log probs to probs.
            return math.exp(log_prob / (transition_ct or 1))

        model_mat = model_data['mat']
        threshold = model_data['thresh']
        return (avg_transition_prob(l, model_mat) < threshold)

    def string_comparison(self, text1, text2, choice='levenshtein_distance'):
        '''
        text1: String Input 1
        text2: String Input 2
        choice: 'levenshtein_distance' or 'damerau_levenshtein_distance' or 'hamming_distance' or 'jaro_distance' or 'jaro_winkler' or 'match_rating_comparison'
        '''
        # https://jellyfish.readthedocs.io/en/latest/comparison.html
        if choice == 'levenshtein_distance':
            return jellyfish.levenshtein_distance(text1, text2)
        elif choice == 'damerau_levenshtein_distance':
            return jellyfish.damerau_levenshtein_distance(text1, text2)
        elif choice == 'hamming_distance':
            return jellyfish.hamming_distance(text1, text2)
        elif choice == 'jaro_distance':
            return jellyfish.jaro_distance(text1, text2)
        elif choice == 'jaro_winkler':
            return jellyfish.jaro_winkler(text1, text2)
        elif choice == 'match_rating_comparison':
            return jellyfish.match_rating_comparison(text1, text2)
        else:
            print("Wrong Choice")
    def competitive_brand_tag(self, text, word_distance=1):
        '''
        :param text: input review string
        :param word_distance: word distance b/w review word and company word (amazon, amzon): helps avoid spell error
        :param print_word: print which company tag is matching
        :return: True (company tag present in review) or False (company tag not present in review)
        '''
        company_tag = []
        with open('src/DictionaryUtils/company_tags.txt', 'r') as fp:
                data = fp.read().lower()
        company_tag = data.split('\n')
        company_tag = set(company_tag)
            # print(self.company_tag)

        input_str = text.split()
        for x in input_str:
            for y in company_tag:
                try:
                    if self.string_comparison(text1=x, text2=y, choice='damerau_levenshtein_distance') <= word_distance:
                        print("Delete for:", x, y)
                        return True
                except:
                    pass
        return False

    def english_swear_check(self, string):
        '''
        input: string
        output: True if text has english proganity False if no profanity
        '''
        english_swear_words = []
        try:
            with open('src/DictionaryUtils/english_profanity_google.txt', 'r') as fp:
                data = fp.read().lower()
            english_swear_words = set(data.split('\n'))
        except:
            print('Warning: english_profanity_google.txt not read')
            pass
        english_swear_words = set(english_swear_words)
        if '' in english_swear_words or ' ' in english_swear_words:
            english_swear_words.pop()

        for word in english_swear_words:
            if word in string.lower():
                return True
        return False

    def main(self,config_path):
        config=read_yaml(config_path)
        local_data_dir=config["source_download_dir"]["data_dir"]
        data_dir= config["source_download_dir"]["data_extract"]
        full_directory = os.path.join(local_data_dir, data_dir)
        data_filename = config["source_download_dir"]["data_file"]
        local_data_filepath = os.path.join(full_directory, data_filename)
        data_preprocess_dir=config["preprocessed"]["data_dir"]
        data_preprocess=config["preprocessed"]["data_processed"]
        preprocess_dir=os.path.join(data_preprocess_dir,data_preprocess)
        create_directories([preprocess_dir])
        data_preprocess_file=config["preprocessed"]["data_file"]
        file_path=os.path.join(preprocess_dir,data_preprocess_file)
        try:
            #Reading the Data
            df=pd.read_csv(local_data_filepath)
            #counting the words in review
            df["review_len"]=df["answer_option"].apply(lambda x:len(x.split()))
            #removing answer_option data which is not in english language
            # 1.Language detection
            logging.info("Step-1 ==> Language Detection started")
            bad_reviews = []
            for indx in df.index:
                review = df.at[indx, 'answer_option']
                try:
                    b = self.language_detection(review)
                    if b == 'hi':
                        bad_reviews.append(indx)
                    elif b == 'en':
                        pass
                except:
                    bad_reviews.append(indx)
            logging.info("Step-1 ==> Language Detection Ended")
            #removing the unwanted data which is not an meaningful information
            logging.info("Step-2 ==> Gibberish Detection Started")
            for indx in df.index:
                review = df.at[indx, 'answer_option']
                if self.gibberish_detection(review):
                    bad_reviews.append(indx)
            logging.info("Step-2 ==> Gibberish Detection Ended")
            logging.info("Step-3 ==> Profanity Detection Started")
            for indx in df.index:
                review = df.at[indx, 'answer_option']
                if self.english_swear_check(review):
                    bad_reviews.append(indx)
            logging.info("Step-3 ==> Profanity Detection Ended")
            logging.info("Step-4 ==> Company Tag Removal Started")
            for indx in df.index:
                review = df.at[indx, 'answer_option']
                if self.competitive_brand_tag(review):
                    bad_reviews.append(indx)
            logging.info("Step-4 ==> Company Tag Removal Ended")
            df=df[~df.index.isin(bad_reviews)].reset_index(drop=True)
            print(df.head())
            df.to_csv(file_path,index=False)
            logging.info("Completed all the Stages in preprocessing data was stored in {}".format(file_path))
        except Exception as e:
            logging.exception(e)
            raise e



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    pre=preprocessing()
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        pre.main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e

