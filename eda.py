import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation

def plot_abv_ibu(data):
    fig, ax = plt.subplots(figsize=(6,6))
    x = data['abv'].values
    y = data['ibu'].values
    ax.scatter(x,y)
    ax.set_title('Hopness vs. Alcohol Percent by Volume')
    ax.set_ylabel('IBU')
    ax.set_xlabel('ABV')
    plt.show()


def plot_missing_values(data):
    fig, ax = plt.subplots(figsize=(6,6))
    data.plot(kind='bar')
    plt.show()

def impute_df(df, algorithm):
    """Returns completed dataframe given an imputation algorithm"""
    return pd.DataFrame(data=algorithm.fit_transform(df), columns=df.columns, index=df.index)

def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))

def plot_new_ibu(data):
    ibu = data['ibu']
    fig, ax = plt.subplots(figsize=(6,6))
    ibu.plot(kind='hist')
    plt.show()


if __name__ == '__main__':
    # Load clean data
    data = pd.read_csv('craft-cans/clean.csv')
    data.drop('Unnamed: 0', axis=1, inplace=True)
    # First basic plot
    plot_abv_ibu(data)
    # Find AVG IBU by style of beer
    avg_ibu_by_style = []
    scope = data['style'].unique()
    for beer in scope:
        avg_ibu = data[data['style']==beer]['ibu'].mean()
        avg_ibu_by_style.append(avg_ibu)

    # Add Column with avg_ibu to associated missing ibu
    # data['avg_ibu'] = 0
    # for beer in data:
    #     scope = data['style'].unique()
    #     for beer_style in scope:
    #         avg_ibu = data[data['style']==beer_style]['ibu'].mean()
    #         if beer_style == beer['style']:
    #             beer['avg_ibu'] = avg_ibu
    #         if beer['ibu'].isnull():
    #             beer['ibu'] = beer['avg_ibu']
    missing = data.isnull().sum()
    # plot_missing_values(missing)
    new_cols = impute_df(data[['abv','ibu']], KNN(k=3))
    vectorizer = CountVectorizer(max_features=150, stop_words='english')
    dtm = vectorizer.fit_transform(data['beer_name'])
    features = vectorizer.get_feature_names()

    # num_topics = 5
    # lda = LatentDirichletAllocation(n_components=num_topics)
    # lda.fit(dtm)
    # find top ten features/words
    # num_top_words = 10
    # display_topics(lda, features, num_top_words)
    plot_new_ibu(new_cols)
