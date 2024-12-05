from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Extract features from the form
            test_weather = {
                'precipitation': float(request.form.get('Precipitation')),
                'temp_max': float(request.form.get('Temperature_max')),
                'temp_min': float(request.form.get('Temperature_min')),
                'wind': float(request.form.get('Wind'))
            }
            
            # Create a DataFrame for prediction
            test_df = pd.DataFrame([test_weather])
            
            # Generate prediction
            prediction = model.predict(test_df)
            prediction_text = f"Predicted CO2 Emission: {prediction[0]}"
        except Exception as e:
            prediction_text = f"An error occurred: {e}"

    # Render the form and pass the prediction result
    return render_template('new.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
