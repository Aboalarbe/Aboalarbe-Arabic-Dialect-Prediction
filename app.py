import json
from flask import Flask, request
import numpy as np
from keras.models import load_model
import txt_preprocessing as pr

app = Flask(__name__)
model = model_new = load_model("models/model.h5")
@app.route("/predict",methods=['GET'])
def predict():

    txt = request.args.get('text')
    cleaned_txt = pr.cleaning_pipeline(txt)
    prepared_txt = pr.prepare_txt(cleaned_txt)
    output = model.predict(prepared_txt)
    labels = ['AE', 'BH', 'DZ', 'EG', 'IQ', 'JO', 'KW', 'LB', 'LY', 'MA', 'OM','PL' ,'QA', 'SA', 'SD', 'SY', 'TN', 'YE']
    result = str(labels[np.argmax(output)])
    return(result)
	
if __name__ == '__main__':
    from waitress import serve
    serve(app, port=5000)	