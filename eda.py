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
    plt.savefig('images/scatter.png')


def plot_missing_values(data):
    fig, ax = plt.subplots(figsize=(6,6))
    data.plot(kind='bar')
    ax.set_title('Missing Values')
    plt.savefig('images/missing_values.png')

def impute_df(df, algorithm):
    """Returns completed dataframe given an imputation algorithm"""
    return pd.DataFrame(data=algorithm.fit_transform(df), columns=df.columns, index=df.index)

def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))

def plot_new_ibu(new_data, old_data):
    ibu_new = new_data['ibu']
    fig, ax = plt.subplots(figsize=(6,6))
    ibu_new.hist(bins=50, label='New')
    old_data.hist(bins=50, label='Old')
    ax.set_title('New IBU values vs. Old IBU values')
    ax.set_xlabel('IBU')
    plt.savefig('images/post_impute.png')


if __name__ == '__main__':
    # Load clean data
    data = pd.read_csv('craft-cans/clean.csv')
    data.drop('Unnamed: 0', axis=1, inplace=True)

    # basic plots
    plot_abv_ibu(data)
    missing = data.isnull().sum()
    plot_missing_values(missing)
    # drop rows with NaN value in ibu and abv column
    data.dropna(subset=['abv', 'ibu'], how='all', inplace=True)
    new_cols = impute_df(data[['abv','ibu']], KNN(k=3))
    plot_new_ibu(new_cols, data['ibu'])
    # Replace columns
    data['abv'] = new_cols['abv']
    data['ibu'] = new_cols['ibu']

    vectorizer = CountVectorizer(max_features=150, stop_words='english')
    dtm = vectorizer.fit_transform(data['beer_name'])
    features = vectorizer.get_feature_names()

    # num_topics = 5
    # lda = LatentDirichletAllocation(n_components=num_topics)
    # lda.fit(dtm)
    # find top ten features/words
    # num_top_words = 10
    # display_topics(lda, features, num_top_words)

    data.to_csv('craft-cans/cleaner.csv')
    plt.show()
