import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from src.topic_modeler import load_data

scaler = StandardScaler

class PipelineHandler(object):
    def __init__(self):
        self.text = None
        self.names = None
        self.beer_features = None
        self.vectorizer = None
        self.dim_reducer = None
        self.X = None
        self.model = None
        self.distance_func = None

    def set_text(self,text, names):
        self.names = names
        self.text = text

    def fit(self):
        pass

    def transform(self):
        matrix = self.vectorizer.transform(self.text)
        self.X = self.dim_reducer.transform(matrix.toarray())

    def predict(self):
        labels = self.model.predict(self.X)
        self.X = np.column_stack((self.X,labels))
        return labels

    def find_closest_beer_names(self):
        pass

    def recommend(self,user_input):
        pass

def no_number_preprocessor(tokens):
    r = re.sub('(\d)+', '', tokens.lower())
    return r

stop_words = ENGLISH_STOP_WORDS.union({'king','german','brau','james',\
'brewery','company','brewing','house','bock','style','scotch','california','oktoberfest',\
'wee','special','english','american','hefeweizen','old','common','gose','NUM'})

if __name__ == '__main__':
    df = load_data()
