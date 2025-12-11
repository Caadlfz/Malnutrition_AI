import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    'Age_Months': [12, 24, 36, 12, 24, 36, 48, 10, 60, 18],
    'Height_cm':  [75, 87, 96, 60, 70, 80, 103, 70, 110, 78],
    'Weight_kg':  [10, 12, 14, 6, 8, 10, 16, 9, 18, 11],
    'Status':     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0] # 0: Healthy, 1: Malnourished
}
df = pd.DataFrame(data)

# 2. PREPARE DATA
X = df[['Age_Months', 'Height_cm', 'Weight_kg']] # Features
y = df['Status']                                 # Labels

# 3. TRAIN MODEL
# We use Random Forest, a common algorithm for classification tasks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. SAVE THE MODEL
# This allows the website to use the AI without retraining it every time
joblib.dump(model, 'malnutrition_model.pkl')
print("Model trained and saved as 'malnutrition_model.pkl'")