from flask import Flask, jsonify, request, render_template
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from src.topic_modeler import TopicModeler, load_data
from src.hard_clustering import recommend_beer, find_closest_beer_names, no_number_preprocessor


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    """Home page that describes the site and explains how it works"""
    # return jsonify({'message':'Hello World!'})
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Route that predicts image upload"""
    pass


@app.route('/recommend', methods=['POST'])
def recommend():
    """Route that recommends a table of beers"""
    # user_preference = request.json # for postman testing
    user_preference = request.form.to_dict(flat=True)
    user_preference['abv'] = float(user_preference['abv']) * 0.01
    prediction = recommend_beer(user_preference,df,vectorizer,pca,km)
    # return jsonify(prediction.to_dict()) # testing json response
    temp = prediction.sort_values('brewery_name').to_dict('records')
    rec_output = []
    for record in temp:
        rec_output.append({
                            'beer_name':record['beer_name'],
                            'abv':round(record['abv']*100,2),
                            'ibu':round(record['ibu'],0),
                            'style':record['style'],
                            'ounces':record['ounces'],
                            'brewery_name':record['brewery_name'],
                            'city':record['city'],
                            'state':record['state']
                        })
    column_names = ['beer_name','abv','ibu','style','ounces',\
                'brewery_name','city','state']
    target = {
        'beer_name':user_preference['beer_name'],
        'style':user_preference['style'],
        'abv':user_preference['abv'] * 100,
        'ibu':user_preference['ibu']
    }
    return render_template('recommend.html',
                            target=target,
                            records=rec_output,
                            colnames=column_names)

if __name__ == '__main__':
    vectorizer = joblib.load('models/hard_vectorizer.pkl')
    pca = joblib.load('models/pca.pkl')
    km = joblib.load('models/km.pkl')
    df = load_data()
    app.run(host='0.0.0.0',port=8080,debug=True)
