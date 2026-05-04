from flask import Flask, request, jsonify
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

# ✅ HOME ROUTE (VERY IMPORTANT FOR RENDER)
@app.route("/")
def home():
    return "TrustGuard AI API running 🚀"

# ✅ PREDICT ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]

    # Transform input
    vector = vectorizer.transform([text])

    # Predict
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    return jsonify({
        "prediction": str(prediction),
        "confidence": float(confidence)
    })


# ✅ RUN APP
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)