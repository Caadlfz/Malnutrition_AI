from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)


model_path = 'malnutrition_model.pkl'


if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print(f"Warning: {model_path} not found. Run train_model.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        data = request.json
        
       
        age = float(data['age'])
        height = float(data['height'])
        weight = float(data['weight'])

        
        if height > 200:  
            return jsonify({'result': "Error: Height is unrealistic (must be < 200cm)"})
        
        
        if weight < 1 or weight > 150:
            return jsonify({'result': "Error: Weight is unrealistic (must be 1kg - 150kg)"})
        
        
        if age < 0:
            return jsonify({'result': "Error: Age cannot be negative"})
        

        
        input_data = pd.DataFrame([[age, height, weight]], 
                                columns=['age_months', 'height_cm', 'weight_kg'])

        
        prediction = model.predict(input_data)[0]
        
        
        if prediction == 0:
            result = "the children is nourished"
        elif prediction == 1:
            result = "moderate malnourished"
        elif prediction == 2:
            result = "severe malnourished"
        else:
            result = "Status Unknown"
        
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 