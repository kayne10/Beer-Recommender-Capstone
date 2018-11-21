import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import GridSearchCV, train_test_split

# pd.options.display.max_columns = 30

stop_words = ENGLISH_STOP_WORDS.union({'king','german','brau','james',\
'brewery','company','brewing','house','bock','style','scotch','california','oktoberfest',\
'wee','special','english','american','hefeweizen','old','common','gose'})

scaler = StandardScaler()

class TopicModeler(object):
    """
    Topic Modeler
    ---------
    model : type of sklearn model to use (currently set up for LDA  but could be extended to NMF and others)
    vectorizer : sklearn vectorizer (currently set up for CountVectorizer but could be extended to use TfIDF)
    distance_func : sklearn.pairwise function for determining distance between documents
    """
    def __init__(self, model, vectorizer, distance_func=cosine_distances):
        self.model = model
        self.text = None
        self.names = None
        self.target_vectors = None
        self.vectorizer = vectorizer
        self.feature_names = None
        self.doc_probs = None
        self.distance_func = distance_func
        self.word_vec = None
        self.other_features = None

    def set_feature_names(self):
        """Sets feature names from vectorizer"""
        self.feature_names = np.array(self.vectorizer.get_feature_names())

    def vectorize(self):
        """Vectorizes corpus"""
        print('Vectorizing...')
        return self.vectorizer.fit_transform(self.text)

    def fit(self, text, names=None):
        """Vectorizes corpus and fits model with it. Beers names are set here as well"""
        self.text = text
        self.word_vec = self.vectorize()
        print('Fitting...')
        self.model.fit(self.word_vec)
        self.set_feature_names()
        self.names = names

    def top_topic_features(self, num_features=6):
        """Takes the n most important features for each topic and indexes against the vectorizer to return the labels associated with those features."""
        # Argsort along rows, reverse the order of the rows, then take all rows up to the number of features
        sorted_topics = self.model.components_.argsort(axis=1)[:, ::-1][:, :num_features]
        # Index the feature names based on the indices of the top most important features
        return self.feature_names[sorted_topics]

    def predict_proba(self, text):
        """Vectorizes and transforms text to give the topic probabilities for each document"""
        # Convert single document to iterable
        if type(text) == str:
            text = [text]
        vec_text = self.vectorizer.transform(text)
        return self.model.transform(vec_text)

    def get_more_features(self, df):
        """Adds two feature columns (abv and ibu values) from beer dataframe"""
        self.other_features = df[['abv','ibu']]
        return self.other_features

    def sort_by_distance(self, doc_index, doc_probs):
        """Finds indices of closest (most similar) beers based on document by probability array"""
        vectors = np.column_stack((doc_probs, self.other_features)) # Append other features to doc_probs with a column_stack
        doc_probs = scaler.fit_transform(vectors)
        # doc_probs = doc_probs[:,-2:] * 2 # weigh features that are more important than others
        distances = self.distance_func(doc_probs[doc_index, np.newaxis], doc_probs)
        return distances.argsort()

    def find_closest_document_titles(self, sorted_indices, num_documents=10):
        """Returns the document beer names of most similar documents. Used for internal dataset."""
        name_array = self.names.iloc[sorted_indices.ravel()][:num_documents]
        return {name_array.iloc[0]: name_array.iloc[1:].tolist()}

    def find_closest_beer_names(self, target_beer, sorted_indices, num_documents=10):
        """Returns beer names that are most similar to a suggested beer"""
        name_array = self.names.iloc[sorted_indices.ravel()][:num_documents]
        return {
            'target_beer': target_beer,
            'recommendations': name_array.tolist()
        }

    def find_beer_idx(self, beer_name):
        """Returns beer name"""
        return self.names[self.names == beer_name].index[0]

    def top_closest_beers(self, beer_name, num_documents=10):
        """Returns most similar articles given article title"""
        doc_index = self.find_beer_idx(beer_name)
        doc_probs = self.predict_proba(self.text)
        beer_similarities = self.sort_by_distance(doc_index, doc_probs)
        return self.find_closest_document_titles(beer_similarities, num_documents)

    def recommend(self, user_input_data):
        """Takes in a users input data that pipelines thru the process of fit -> transform -> recommend"""
        text = user_input_data['beer_name'] + ' ' + user_input_data['style']
        vectorize_text = self.vectorizer.transform(text.split('  '))
        input_doc_probs = self.model.transform(vectorize_text)
        beer_features = np.column_stack((user_input_data['abv'], user_input_data['ibu']))
        input_vector = np.column_stack((input_doc_probs, beer_features))
        all_doc_probs = self.predict_proba(self.text)
        all_doc_vectors = np.column_stack((all_doc_probs, self.other_features))
        all_doc_vectors = scaler.fit_transform(all_doc_vectors)
        input_vector = scaler.transform(input_vector)
        # input_vector = input_doc_probs[:,-2:] * 2
        # all_doc_vectors = all_doc_vectors[:,-2:] * 2
        distances = self.distance_func(input_vector, all_doc_vectors)
        sorted_distances = np.argsort(distances)
        return self.find_closest_beer_names(user_input_data['beer_name'],sorted_distances,10)

    def grid_search_lda(self, params):
        """Performs grid search optimizing log-loss using topic model"""
        # Create holdout set to check log loss on unseen data
        X_train, X_test = train_test_split(self.word_vec, test_size=0.25)
        lda_cv = GridSearchCV(self.model, param_grid=params, n_jobs=-1,  verbose=1)
        lda_cv.fit(X_train)

        print(pd.DataFrame.from_dict(lda_cv.cv_results_))
        print('Test Score:', lda_cv.score(X_test))
        return lda_cv

    def load_model(self, filepath):
        """Load topic model object"""
        self.model = joblib.load(filepath)

    def load_vectorizer(self, filepath):
        """Load vectorizer object"""
        self.tf_vectorizer = joblib.load(filepath)

    def save_model(self, filepath):
        """Save topic model object"""
        joblib.dump(self.model, filepath)

    def save_vectorizer(self, filepath):
        """Save vectorizer object"""
        joblib.dump(self.vectorizer, filepath)

def load_data():
    """Loads data"""
    df = pd.read_csv('craft-cans/cleaner.csv')
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df

def handle_input(fav_beer, fav_style, abv_pref, ibu_pref):
    """Takes a users preference of beer"""
    return {
            'beer_name':fav_beer,
            'style':fav_style,
            'abv':abv_pref,
            'ibu':ibu_pref
    }

def main():
    df = load_data() #add this make a sample .sample(100).reset_index(drop=True)
    # best params so far: n_components=7, doc_topic_prior=1/8, topic_word_prior=0.4
    lda = LatentDirichletAllocation(n_components=7, learning_offset=50., verbose=1,
                                    doc_topic_prior=1/8, topic_word_prior=0.4, n_jobs=-1, learning_method='online')
    tf_vectorizer = CountVectorizer(max_df=0.85, min_df=2, max_features=850, stop_words=stop_words)
    tm = TopicModeler(lda, tf_vectorizer)
    tm.fit(df['beer_name'] + ' ' +  df['style'], names=df['beer_name'])
    tm.get_more_features(df)
    print(tm.top_topic_features())

    user_preference = handle_input('Voodoo Ranger Imperial IPA','Imperial IPA',0.09,70.0)
    # print(tm.recommend(user_preference))

    print(tm.top_closest_beers(df.beer_name.iloc[17],10)) # imperial ipa
    print(tm.top_closest_beers(df.beer_name.iloc[1090],10)) # cider
    print(tm.top_closest_beers(df.beer_name.iloc[179],10)) # irish stout
    print(tm.top_closest_beers('Iron Mike Pale Ale',10))
    print("\n")
    print(tm.recommend(user_preference))
    response = tm.recommend(user_preference)

    # Commented out because this can take a really long time!
    # params = {"n_components": [3, 5, 7, 9]}
    # grid_search = tm.grid_search_lda(params)
    # joblib.dump(grid_search, 'grid_search.joblib')
    # A grid search across many values yielded these values for optimizing log loss:
    # {'doc_topic_prior': 0.25, 'n_components': 5, 'topic_word_prior': 0.5}
    tm.save_vectorizer('vectorizer.pkl')
    tm.save_model('model.pkl')

if __name__ == '__main__':
    main()
