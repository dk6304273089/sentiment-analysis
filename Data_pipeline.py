from src.stage_02_preprocessing import preprocessing
from src.stage_03_feature_engineering import feature_engineering
from src.stage_04_model_training import model_training
import argparse
import pandas as pd
from collections import Counter
from statistics import mean
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab,CatTargetDriftTab
bad_reviews = set()
language_error = set()
gibberish = set()
swear = set()
company_tag = set()
pre = preprocessing()
fr=feature_engineering()
mt=model_training()
class test:
    def main(self,file_path):
        try:
            df=pd.read_csv(file_path)
            df["review_len"] = df["answer_option"].apply(lambda x: len(x.split()))
            bad_reviews = []
            for indx in df.index:
                review = df.at[indx, 'answer_option']
                try:
                    b = pre.language_detection(review)
                    if b == 'hi':
                        bad_reviews.append(indx)
                    elif b == 'en':
                        pass
                except:
                    language_error.add(indx)

                if pre.gibberish_detection(review):
                    gibberish.add(indx)

                if pre.english_swear_check(review):
                    swear.add(indx)

                if pre.competitive_brand_tag(review):
                    company_tag.add(indx)
            print(
                "Number of Bad Reviews for Language Error: {} \n Number of Bad Reviews for Gibberish: {} \n Number of Bad Reviews for Swear: {} \n Number of Bad Reviews for Competitive Brand: {}".format(
                    len(language_error), len(gibberish), len(swear), len(company_tag)))
            print("DELETED REVIEWS: \n", df[df.index.isin(bad_reviews)])
            df = df[~df.index.isin(bad_reviews)].reset_index(drop=True)
            df = df.sort_values(by=['product'], ignore_index=True)
            df['review_len'] = df['answer_option'].apply(lambda x: len(x.split()))
            df['Noun_Strength'] = 0.0
            df['Review_Polarity'] = 0.0
            df['Review_Subjectivity'] = 0.0
            df['Review_Complexity'] = 0.0
            df['Service_Tagger'] = 0.0
            df['Compound_Score'] = 0.0
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
                    df.at[indx, 'Review_Polarity'] = fr.polarity_sentiment(review)
                    df.at[indx, 'Review_Subjectivity'] = fr.subjectivity_sentiment(review)
                    df.at[indx, 'Service_Tagger'] = fr.service_tag(review)
                    df.at[indx, 'Compound_Score'] = fr.slang_emoji_polarity_compoundscore(review)
                    df.at[indx, 'Review_Complexity'] = float(len(set(review.split()))) / float(len(unique_bag))
                df.loc[df['product'] == product, 'Noun_Strength'] = fr.noun_score(data['answer_option'].values).values
            product_list = df['product'].unique()
            df['win'] = 0
            df['lose'] = 0
            df['review_score'] = 0.0
            df.reset_index(inplace=True, drop=True)

            for product in product_list:
                data = df[df['product'] == product]
                for indx in data.index:
                    review = df.loc[indx, ['review_len', 'Noun_Strength', 'Review_Polarity', 'Review_Subjectivity', 'Review_Complexity', 'Service_Tagger', 'Compound_Score']]
                    review['j'] = 'jn'
                    C = pd.DataFrame([review])
                    D = data[data.index != indx].loc[:, ['review_len', 'Noun_Strength', 'Review_Polarity', 'Review_Subjectivity', 'Review_Complexity', 'Service_Tagger', 'Compound_Score']]
                    D['j'] = 'jn'
                    E = pd.merge(C, D, how='outer', on='j')
                    score = mt.score_giver(C, D)
                    df.at[indx, 'win'] = 0 if score.get(1) is None else score.get(1)
                    df.at[indx, 'lose'] = 0 if score.get(0) is None else score.get(0)
                    df.at[indx, 'review_score'] = float(0 if score.get(1) is None else score.get(1)) / len(data) * 1.0
                print(" Reviews of Product: {} Ranked".format(product))
            if parsed_args.testing == 'True':
                data_split = pd.crosstab(df['product'], df['label'])
                r_accuracy = []
                for product in product_list:
                    x = data_split[data_split.index == product][1][0]
                    number_of_1_in_x = Counter(df[df['product'] == product].iloc[:x, ]['label']).get(1)
                    rank_accuracy = float(number_of_1_in_x * 1.0 / x * 1.0)
                    print("Product: {} | Rank Accuracy: {}".format(product, rank_accuracy))
                    r_accuracy.append(rank_accuracy)
                print("TEST DATA: Mean Rank Accuracy: {}".format(mean(r_accuracy)))
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
            print(data[['product', 'answer_option', 'review_score','rank']])
            data[['product', 'answer_option', 'review_score','rank']].to_csv('data/test_ranked_output.csv', index=False)
            data_drift_report = Dashboard(tabs=[DataDriftTab()])
            df=pd.read_csv("data/feature_processed/feature.csv")
            data=data[["Noun_Strength","Review_Polarity","Review_Subjectivity","Review_Complexity","Service_Tagger","Compound_Score"]]
            df=df[["Noun_Strength","Review_Polarity","Review_Subjectivity","Review_Complexity","Service_Tagger","Compound_Score"]]
            data_drift_report.calculate(df, data, column_mapping=None)
            data_drift_report.save("report/my_report.html")
        except Exception as e:
            raise e

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--file_name", "-f", default="data/test.csv",help="Data path")
    args.add_argument("--testing","-t",type=str, default='False')
    parsed_args = args.parse_args()
    pr=test()
    try:
        pr.main(file_path=parsed_args.file_name)
    except Exception as e:
        raise  e
