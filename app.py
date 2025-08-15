from flask import Flask, request, jsonify
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data'] # Expects json like {"data": ["this is a good comment", "this is bad"]}
        
        pipeline = PredictPipeline()
        predictions = pipeline.predict(data)
        
        # Convert numpy array to list for JSON serialization
        # return jsonify(list(predictions))
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
