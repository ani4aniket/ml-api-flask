from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# load the model from disk
regressor = pickle.load(open("regressor.model", 'rb'))

sc_X = pickle.load(open("scaler_X.model", 'rb'))

sc_y = pickle.load(open("scaler_y.model", 'rb'))

app = Flask(__name__)

@app.route('/predict/<exp>')
def predict(exp):
    sample_input = np.array([exp]).reshape(-1, 1)
    sample_output = regressor.predict(sc_X.transform(sample_input))
    return str((sc_y.inverse_transform(sample_output))[0][0])

@app.route("/", methods=['GET', 'POST'])
def hello():
    
    exp = -1

    try:
        exp = request.form['experience']
    except:
        pass
    
    ans = 0
    
    if exp != -1:
        ans = predict(exp)
    
    return render_template('index.html', ans = ans)

if __name__ == "__main__":
    app.run()