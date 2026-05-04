from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return "TrustGuard AI API running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    return jsonify({
        "prediction": prediction,
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)