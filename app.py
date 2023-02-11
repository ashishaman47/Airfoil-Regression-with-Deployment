import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# start a flask app
app = Flask(__name__)

# load the pickle file --> ML model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    # Render Home Page
    return render_template('home.html')



# creating an api
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # capture the json data coming from postman
    # Order of input data should be in same order in which you have trained your data
    data = request.json['data']
    print(data)

    # converting input data into 2D to send as input to model
    new_data = [list(data.values())]

    # performing prediction from model
    output = model.predict(new_data)[0]

    return jsonify(output)


@app.route('/predict', methods=['POST'])
def predict():
    # getting the data from the html form
    data = [float(x) for x in request.form.values()]
    
    # converting the data into 2D array to give as input value in model
    final_val = [np.array(data)]
    print(data)
    print(final_val)

    # prediction of model
    output = model.predict(final_val)[0]
    print(output)

    return render_template('home.html', prediction_text='Airfoil Pressure is : {}'.format(output))


# point of execution of entire code
if __name__ == '__main__':
    app.run(debug=True)
