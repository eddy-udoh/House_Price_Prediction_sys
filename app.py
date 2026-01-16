from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
app = Flask(__name__)

# Load the trained model and the scaler
# (Make sure you ran create_model.py first!)
try:
    model = load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
except:
    print("ERROR: model.h5 or scaler.pkl not found. Run create_model.py first.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            # Get values from the HTML form
            # These match the feature names in the California dataset
            med_inc = float(request.form['MedInc'])
            house_age = float(request.form['HouseAge'])
            ave_rooms = float(request.form['AveRooms'])
            ave_bedrms = float(request.form['AveBedrms'])
            population = float(request.form['Population'])
            ave_occup = float(request.form['AveOccup'])
            latitude = float(request.form['Latitude'])
            longitude = float(request.form['Longitude'])

            # Prepare features for the model
            features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, 
                                  population, ave_occup, latitude, longitude]])
            
            # Scale the features (MUST do this because we did it in training)
            features_scaled = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features_scaled)
            price = prediction[0][0] * 100000 # Convert units to dollars

            prediction_text = f"Estimated House Price: ${price:,.2f}"
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)