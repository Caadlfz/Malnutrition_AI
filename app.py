from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('malnutrition_model2.pkl')

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

        # Create a DataFrame with the exact column names used in training
        input_data = pd.DataFrame([[age, height, weight]], 
                                columns=['age_months', 'height_cm', 'weight_kg'])

        # Make prediction
        print(age,height,weight)
        prediction = model.predict(input_data)[0]
        print(f"Prediction raw output: {prediction}")
        
        # Interpret result based on our new mapping
        if prediction == 0:
            result = "Healthy Growth Range"
        elif prediction == 1:
            result = "Potential Moderate Malnutrition"
        elif prediction == 2:
            result = "Potential Severe Malnutrition"
        else:
            result = "Status Unknown"
        
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)