import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model =joblib.load(open('results/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [i for i in request.form.values()]

    dictionary = {'Female': 0, 'Male': 1, 'No': 0, 'Yes': 1, '0': 0, '1':1, '2':2, '3+':3, \
            'Graduate' : 0, 'Not Graduate' : 1, 'Rural' : 0, 'Urban' : 1, 'Semiurban' : 2}

    #Convert corresponding categorical feature to numeric values as inputs for the model
    numeric_features =  [dictionary.get(item, item) for item in features] 
    print(numeric_features)
    #convert array to float datatype
    numeric_features_float = [np.array(numeric_features).astype(np.float64)]

    #predict loan status
    prediction = model.predict(numeric_features_float)
    output = prediction[0]
    if output == 'Y':
        output = "Congratulations, your Loan application Status is Approved!"
    else:
        output = 'Sorry, your Loan application status is Rejected!'

    return render_template('index.html', prediction_text=output)


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)