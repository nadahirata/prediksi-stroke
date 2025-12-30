from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# load model
model = joblib.load("model_stroke.pkl")

@app.route("/")
def home():
    return render_template("input_pasien.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    input_data = {
        "gender": int(data["gender"]),
        "age": float(data["age"]),
        "hypertension": int(data["hypertension"]),
        "heart_disease": int(data["heart_disease"]),
        "avg_glucose_level": float(data["avg_glucose_level"]),
        "bmi": float(data["bmi"])
    }

    df = pd.DataFrame([input_data])
    prob = model.predict_proba(df)[0][1]

    hasil = "Stroke" if prob > 0.5 else "Tidak Stroke"

    return jsonify({
        "hasil": hasil,
        "probabilitas": round(prob*100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
