import pandas as pd
from nltk.metrics import edit_distance
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
import numpy as np
from copy import deepcopy
import os
import yaml
import logging
import time
from textblob import TextBlob, Word
import spacy
nlp = spacy.load("en_core_web_sm")
import argparse
import logging
from src.utils.common import read_yaml,create_directories

STAGE = "Stage 03 Feature Engineering Step"
logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

class feature_engineering:
    def polarity_sentiment(self, text):
        '''
        input: string
        output: value between -1 to +1
        '''
        blob = TextBlob(text)
        return (blob.sentiment.polarity)

    def subjectivity_sentiment(self, text):
        '''
        input: string
        output: 0 to 1
        '''
        blob = TextBlob(text)
        return (blob.sentiment.subjectivity)

    def service_tag(self, text):
        '''
        text: string input
        output: 0 or 1
        '''
        tagger = []
        try:
            with open('src/DictionaryUtils/service_tagger.txt', 'r') as fp:
                data = fp.read().lower()
            tagger = set(data.split('\n'))
        except:
            print('Warning: Service_tagger.txt not read')
            pass
            tagger = set(tagger)

            if '' in tagger or ' ' in tagger:
                tagger.pop()

        k = text.split()
        for w in k:
            for wrd in tagger:
                x = edit_distance(w.lower(), wrd)
                if x <= 1:
                    return 1
        return 0

    def slang_emoji_polarity_compoundscore(self, text):
        '''
        Input: Text
        Output:
        (-0.5 to +0.5): Neural
        (-inf to -0.5): Negative
        (+0.5 to +inf): Positive
        '''
        analyzer=SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)['compound']

    def corpus_stem_lemma(self, corpus):
        '''
        Input: Corpus(List of Strings)
        Output: A lemmatized and stemmed Corpus
        '''
        for i in range(len(corpus)):
            doc = nlp(corpus[i])
            corpus[i] = " ".join([lemmatizer.lemmatize(token.lemma_) for token in doc if
                                  token.is_stop == False and token.is_punct == False and token.is_alpha == True])
            # print(temp[i])
        return corpus
    def noun_score(self, corpus):
        '''
        TFIDF_NOUN_SCORE = Sum of TFIDF OF NOUN in a Review / Sum of TFIDF of all words in that review
        :param corpus:
        :return:
        '''
        noun_tag = []
        for review in corpus:
            doc = nlp(review)
            noun_tag.append([lemmatizer.lemmatize(token.lemma_) for token in doc if
                             token.pos_ == "NOUN" and token.is_stop == False and token.is_punct == False and token.is_alpha == True])

        corpus = self.corpus_stem_lemma(corpus)

        tfidf = TfidfVectorizer()

        features = tfidf.fit_transform(corpus)
        df_tfidf = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())
        df_tfidf['sum'] = df_tfidf.sum(axis=1)

        df_tfidf['noun_sum'] = 0.0
        df_tfidf['tfidf_score'] = 0.0

        for i in range(len(noun_tag)):
            sm = 0.0
            for q in noun_tag[i]:
                if q in df_tfidf.columns:
                    sm += df_tfidf[q][i]
            df_tfidf.at[i, 'noun_sum'] = sm
            if df_tfidf.at[i, 'sum'] == 0.0:
                df_tfidf.at[i, 'tfidf_score'] = 0.0
                continue
            df_tfidf.at[i, 'tfidf_score'] = float(df_tfidf.at[i, 'noun_sum'] / df_tfidf.at[i, 'sum'])

        return df_tfidf['tfidf_score']
    def main(self,config_path):
        try:
            config = read_yaml(config_path)
            data_preprocess_dir = config["preprocessed"]["data_dir"]
            data_preprocess = config["preprocessed"]["data_processed"]
            preprocess_dir = os.path.join(data_preprocess_dir, data_preprocess)
            data_preprocess_file = config["preprocessed"]["data_file"]
            file_path = os.path.join(preprocess_dir, data_preprocess_file)
            feature_preprocess_dir=config["featured"]["feature_dir"]
            feature_preprocess_file=config["featured"]["feature_processed"]
            feature_dir=os.path.join(feature_preprocess_dir,feature_preprocess_file)
            create_directories([feature_dir])
            feature_file_path=config["featured"]["feature_file"]
            feature_file=os.path.join(feature_dir,feature_file_path)
            df=pd.read_csv(file_path)
            df['Noun_Strength'] = 0.0
            df['Review_Polarity'] = 0.0
            df['Review_Subjectivity'] = 0.0
            df['Review_Complexity'] = 0.0
            df['Service_Tagger'] = 0.0
            df['Compound_Score'] = 0.0
            logging.info(" Added New Columns {Noun Strength,Review Polarity,Review Subjectivity,Review Complexity,Service Tagger,Compound Score}")
            product_list = df['product'].unique()
            for product in product_list:
                data = df[df['product'] == product]
                unique_bag = set()
                for review in data['answer_option']:
                    review = review.lower()
                    words = review.split()
                    unique_bag = unique_bag.union(set(words))
                for indx in data.index:
                    review = data.at[indx, 'answer_option']
                    df.at[indx, 'Review_Polarity'] = self.polarity_sentiment(review)
                    df.at[indx, 'Review_Subjectivity'] = self.subjectivity_sentiment(review)
                    df.at[indx, 'Service_Tagger'] = self.service_tag(review)
                    df.at[indx, 'Compound_Score'] = self.slang_emoji_polarity_compoundscore(review)
                    df.at[indx, 'Review_Complexity'] = float(len(set(review.split()))) / float(len(unique_bag))
                df.loc[df['product'] == product, 'Noun_Strength'] = self.noun_score(data['answer_option'].values).values
            df.to_csv(feature_file,index=False)
            logging.info("Successfully saved the file in {}".format(feature_file))
        except Exception as e:
            logging.exception(e)
            raise e

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    pre=feature_engineering()
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        pre.main(config_path=parsed_args.config)
        logging.info(f">>>>>> stage {STAGE} completed!<<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e