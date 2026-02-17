from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model safely
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    is_female = int(request.form["is_female"])
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    is_smoker = int(request.form["is_smoker"])
    region_southeast = int(request.form["region_southeast"])
    bmi_category_obesity = int(request.form["bmi_category_obesity"])

    features = np.array([[age, is_female, bmi, children,is_smoker, region_southeast,bmi_category_obesity]])

    prediction_log = model.predict(features)[0]
    prediction = np.expm1(prediction_log)

    # Instead of rendering directly, redirect with query parameter
    return redirect(url_for("result", cost=round(prediction, 2)))

@app.route("/result")
def result():
    cost = request.args.get("cost")
    prediction_text = f"Estimated Insurance Cost: ₹ {cost}"
    return render_template("index.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)