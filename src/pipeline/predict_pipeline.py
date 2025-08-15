import pickle
import pandas as pd

class PredictPipeline:
    def __init__(self):
        self.vectorizer_path = "artifacts/transformed_data/vectorizer.pkl"
        self.model_path = "artifacts/trained_model/model.pkl"
        
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, text_data):
        transformed_data = self.vectorizer.transform(text_data)
        predictions = self.model.predict(transformed_data)
        return predictions
