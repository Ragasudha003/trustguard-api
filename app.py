from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# =========================
# LOAD MODEL & VECTORIZER
# =========================
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)


# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return "TrustGuard AI API running 🚀"


# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        # Transform input
        vector = vectorizer.transform([text])

        # Prediction
        prediction_num = model.predict(vector)[0]
        confidence = max(model.predict_proba(vector)[0])

        # Convert to readable output
        prediction = "spam" if prediction_num == 1 else "ham"

        return jsonify({
            "prediction": prediction,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# RUN APP (FOR LOCAL + RENDER)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)