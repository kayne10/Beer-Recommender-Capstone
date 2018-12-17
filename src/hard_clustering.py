import numpy as np
import pandas as pd
import re
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
#import matplotlib.pyplot as plt
import seaborn as sns
from src.topic_modeler import load_data, handle_input

sns.set()

def no_number_preprocessor(tokens):
    r = re.sub('(\d)+', '', tokens.lower())
    return r

stop_words = ENGLISH_STOP_WORDS.union({'king','german','brau','james',\
'brewery','company','brewing','house','bock','style','scotch','california','oktoberfest',\
'wee','special','english','american','hefeweizen','old','common','gose','NUM'})

scaler = StandardScaler()

def find_closest_beer_names(target_beer, df, sorted_indices, num_documents=10):
    """Returns beer names that are most similar to a suggested beer"""
    name_array = df['beer_name'].iloc[sorted_indices.ravel()][:num_documents]
    # return {
    #     'target_beer': target_beer,
    #     'recommendations': name_array.tolist()
    # }
    return df[df['beer_name'].isin(name_array.tolist())]

def recommend_beer(user_input, df, vectorizer, dim_reducer, model):
    text = user_input['beer_name'] + ' ' + user_input['style']
    all_text = df['beer_name'] + ' ' + df['style']

    input_matrix = vectorizer.transform(text.split('  '))
    matrix = vectorizer.transform(all_text)

    input_vector = dim_reducer.transform(input_matrix.toarray())
    all_doc_vectors = dim_reducer.transform(matrix.toarray())

    cluster = model.predict(input_vector)
    input_vector = np.column_stack((input_vector, cluster))
    all_doc_vectors = np.column_stack((all_doc_vectors,model.labels_))

    beer_features = np.column_stack((user_input['abv'], user_input['ibu']))
    input_vector = np.column_stack((input_vector, beer_features))
    all_doc_vectors = np.column_stack((all_doc_vectors, df[['abv','ibu']]))
    all_doc_vectors = scaler.fit_transform(all_doc_vectors)
    input_vector = scaler.transform(input_vector)

    distances = cosine_distances(input_vector, all_doc_vectors)
    sorted_indices = np.argsort(distances)
    return find_closest_beer_names(user_input['beer_name'],df,sorted_indices,5)

def plot_elbow(km,X):
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('images/elbow.png')

def plot_explained_variance(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance');
    plt.title('PCA Explanation of Variance')
    plt.savefig('images/explained_variance.png')


def plot_scattered_ibu_abv(results):
    colormap = {0:'red', 1:'green',2: 'blue',3:'purple',4:'orange',5:'yellow',6:'brown'}
    colors = results.apply(lambda row: colormap[row.category], axis=1)
    ax = results.plot(kind='scatter', x='abv', y='ibu', alpha=0.1, s=50, c=colors)
    ax.set_xlabel("ABV")
    ax.set_ylabel("IBU")
    plt.savefig('images/scattered_clusters.png')


if __name__ == '__main__':
    df = load_data() #add this make a sample .sample(100).reset_index(drop=True)

    tf_vectorizer = CountVectorizer(max_df=0.85, min_df=2, max_features=1000,
                                    preprocessor=no_number_preprocessor, stop_words=stop_words)
    pca = PCA(n_components=20, whiten=True)
    n_clusters = 7
    km = KMeans(n_clusters=n_clusters)

    # fit vectorizer and models
    matrix = tf_vectorizer.fit_transform(df['beer_name'] + ' ' + df['style'])
    X2 = pca.fit_transform(matrix.toarray())
    km.fit(X2)

    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tf_vectorizer.get_feature_names()
    for i in range(n_clusters):
        top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
        print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))

    # plot_elbow(km,X2)
    # plot_explained_variance(pca)

    new_df = pd.DataFrame(matrix.toarray(), columns=tf_vectorizer.get_feature_names())
    results = pd.DataFrame({
        'beer_name': df['beer_name'].values,
        'style': df['style'].values,
        'abv': df['abv'].values,
        'ibu': df['ibu'].values,
        'category': km.labels_
    })
    # plot_scattered_ibu_abv(results)
    summarized_results = results.groupby('category').agg({'abv':'mean','ibu':'mean'})

    user_preference = handle_input('Voodoo Ranger Imperial IPA','Imperial IPA',0.09,70.0)
    recs = recommend_beer(user_preference,df,tf_vectorizer,pca,km)
    print(recs)

    # save vectors and models
    # joblib.dump(tf_vectorizer, 'models/hard_vectorizer.pkl')
    # joblib.dump(pca, 'models/pca.pkl')
    # joblib.dump(km, 'models/km.pkl')
