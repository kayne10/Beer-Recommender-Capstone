from flask import Flask, jsonify, request, render_template, send_file, redirect
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from src.topic_modeler import TopicModeler, load_data
from src.hard_clustering import recommend_beer, find_closest_beer_names, no_number_preprocessor
import PIL
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import boto3
import configparser

config = configparser.ConfigParser()
config.read('src/config.ini')

s3 = boto3.client(
    "s3",
    aws_access_key_id=config['S3']['ACCESS_KEY_ID'],
    aws_secret_access_key=config['S3']['SECRET_ACCESS_KEY']
)

def upload_file_to_s3(file, dest, acl="public-read"):

    try:

        s3.upload_fileobj(
            file,
            dest,
            'capstone/uploads/{}'.format(file.filename),
            ExtraArgs={
                "ACL": acl,
                "ContentType": file.content_type
            }
        )
        return file.filename

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Something Happened: ", e)
        return e


app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET'])
def home():
    """Home page that describes the site and explains how it works"""
    # return jsonify({'message':'Hello World!'})
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Route that predicts image upload"""
    img = request.files['beer_image']
    img = image.load_img(img, target_size=(200,200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # Use the loaded model to generate a prediction.
    pred = model.predict(img_tensor)
    label = np.argmax(pred)
    prediction = {'label':int(label)}
    #Upload to S3 for future training
    img = request.files['beer_image']
    output = upload_file_to_s3(img,dest,acl="public-read")
    # return jsonify(prediction)
    return render_template('predict.html',prediction=prediction,upload=output) #render uploaded file too


@app.route('/recommend', methods=['POST'])
def recommend():
    """Route that recommends a table of beers"""
    # user_preference = request.json # for postman testing
    user_preference = request.form.to_dict(flat=True)
    user_preference['abv'] = float(user_preference['abv']) * 0.01
    num_documents = int(user_preference['num_docs'])
    prediction = recommend_beer(user_preference,df,vectorizer,pca,km,num_documents)
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
        'abv':round(user_preference['abv'] * 100,1),
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
    model = load_model('models/transfer_test_two.hdf5')
    model._make_predict_function()
    df = load_data()
    dest = config['S3']['BUCKET_NAME']
    app.run(host='0.0.0.0',port=8080,debug=True)
    # app.run(host='0.0.0.0',port=8080)
