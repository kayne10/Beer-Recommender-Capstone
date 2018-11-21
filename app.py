from flask import Flask, jsonify, request, render_template
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from topic_modeler import TopicModeler, load_data


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # return jsonify({'message':'Hello World!'})
    return render_template('home.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    # user_preference = request.json
    user_preference = request.form.to_dict(flat=True)
    # edit abv value to be a decimal
    user_preference['abv'] = float(user_preference['abv'])
    user_preference['abv'] = user_preference['abv'] * 0.01
    prediction = tm.recommend(user_preference)
    return jsonify(prediction)

if __name__ == '__main__':
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    tm = TopicModeler(model, vectorizer)
    df = load_data()
    # fit model here instead
    tm.fit(df['beer_name'] + ' ' + df['style'], names=df['beer_name'])
    # get more features here instead as well
    tm.get_more_features(df)
    app.run(port=8080)
