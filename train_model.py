import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. LOAD DATA
# Load the specific CSV file uploaded
df = pd.read_csv('malnutrition_data_Children.csv')

# 2. PREPARE DATA
# Features: We select the columns that match the user input
X = df[['age_months', 'height_cm', 'weight_kg']]

# Labels: Map the text status to numbers for the AI to understand
# normal -> 0
# moderate -> 1
# severe -> 2
label_mapping = {'normal': 0, 'moderate': 1, 'severe': 2}
y = df['nutrition_status'].map(label_mapping)

# 3. TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 4. SAVE THE MODEL
joblib.dump(model, 'malnutrition_model2.pkl')
print("Model trained with new dataset and saved as 'malnutrition_model.pkl'")