from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('wine_quality_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predict_request = [data['features']]
    prediction = model.predict(predict_request)

    # Take the first (and only) prediction from the output
    output = int(prediction[0])
    return jsonify(quality=output)

if __name__ == '__main__':
    app.run(debug=True)
