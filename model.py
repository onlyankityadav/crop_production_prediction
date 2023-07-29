from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
label_encoding_info = {
    "District": {
        "Jodhpur": 5,
        "Kota": 6,
        "Jaipur": 4,
        "Hanumangarh": 3,
        "Sri Ganganagar": 8,
        "Udaipur": 9,
        "Bhilwara": 2,
        "Alwar": 1,
        "Nagaur": 7,
        "Ajmer": 0,
    },
    "Crop": {
        "Wheat": 22,
        "Gram": 10,
        "Coriander": 4,
        "Citrus": 3,
        "Cotton": 5,
        "Guava": 11,
        "Garlic": 9,
        "Mustard": 14,
        "Fenugreek": 8,
        "Maize": 12,
        "Fennel": 7,
        "Bajra": 0,
        "Oilseeds": 15,
        "Opium": 17,
        "Pomegranate": 18,
        "Cumin": 6,
        "Chilli": 2,
        "Tomato": 21,
        "Sugarcane": 20,
        "Barley": 1,
        "Onion": 16,
        "Pulses": 19,
        "Mango": 13,
    },
    "Season": {"Kharif": 0, "Rabi": 1},
}
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_prod():
    user_input = ["District", "Crop", "Season", "Area", "Yield"]
    decoded_input = {}

    for i in user_input[:3]:
        encoding_map = label_encoding_info[i]
        decoded_value = encoding_map[request.form.get(i)]
        decoded_input[i] = decoded_value

    data = [decoded_input[i] for i in user_input[:3]]
    data.extend(float(request.form.get(i)) for i in user_input[3:])

    district, crop, season = (
        request.form.get("District"),
        request.form.get("Crop"),
        request.form.get("Season"),
    )
    area, yield1 = float(request.form.get("Area")), float(request.form.get("Yield"))

    result = model.predict(np.array(data).reshape(1, -1))

    result = float(result[0])
    return render_template(
        "forecast.html",
        district=district,
        crop=crop,
        season=season,
        area=area,
        yield1=yield1,
        result=result,
    )


@app.route("/test_again", methods=["POST"])
def test_again():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
