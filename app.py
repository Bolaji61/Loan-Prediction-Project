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
    int_features = [i for i in request.form.values()]
    int_features = np.array(int_features).astype(np.float64)
    final_features = [np.array(int_features)]
    print(int_features)
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    output = prediction[0]
    

    return render_template('index.html', prediction_text='Loan Status is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)