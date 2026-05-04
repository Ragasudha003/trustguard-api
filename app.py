from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# =========================
# LOAD MODEL & VECTORIZER
# =========================
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    print("✅ Model and vectorizer loaded successfully")

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

        # Validate input
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        # Transform text
        vector = vectorizer.transform([text])

        # Predict
        prediction_num = model.predict(vector)[0]
        confidence = max(model.predict_proba(vector)[0])

        # Convert output
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
# RUN APP (Render compatible)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)