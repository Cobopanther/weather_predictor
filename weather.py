from flask import Flask, render_template, request
from datetime import datetime, timedelta
import joblib
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        date = request.form['date']
        try:
            input_date = datetime.strptime(date, "%Y-%m-%d")
            model = joblib.load(os.path.join(os.path.dirname(__file__), 'weather_model.pkl'))
            label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), 'label_encoder.pkl'))
            # 7-day forecast: 3 days before, selected, 3 days after
            forecast = []
            for offset in range(-3, 4):
                day = input_date + timedelta(days=offset)
                features = np.array([[day.year, day.month, day.day, day.weekday()]])
                pred = model.predict(features)[0]
                cond_decoded = label_encoder.inverse_transform([int(round(pred[6]))])[0]
                will_rain = "Yes" if round(pred[7]) == 1 else "No"
                forecast.append({
                    "date": day.strftime('%d-%m-%Y'),
                    "temp": f"{pred[0]:.2f}",
                    "tempmax": f"{pred[1]:.2f}",
                    "tempmin": f"{pred[2]:.2f}",
                    "humidity": f"{pred[3]:.2f}",
                    "windspeed": f"{pred[4]:.2f}",
                    "precip": f"{pred[5]:.2f}",
                    "condition": cond_decoded,
                    "will_rain": will_rain
                })
            # The selected day is the 4th in the list
            prediction_dict = forecast[3]
            return render_template('predict.html', prediction=prediction_dict, forecast=forecast)
        except Exception as e:
            return render_template('predict.html', prediction=None, error=f"Error: {str(e)}")
    return render_template('predict.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
