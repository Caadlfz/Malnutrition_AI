from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('malnutrition_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the frontend
        data = request.json
        age = float(data['age'])
        height = float(data['height'])
        weight = float(data['weight'])

        # Create a DataFrame for prediction (must match training structure)
        input_data = pd.DataFrame([[age, height, weight]], 
                                columns=['Age_Months', 'Height_cm', 'Weight_kg'])

        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Interpret result
        result = "Potential Malnutrition Detected" if prediction == 1 else "Healthy Growth Range"
        
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)