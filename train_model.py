import pandas as pd
from zipfile import ZipFile
from io import BytesIO
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# URL of the ZIP file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

# Download the ZIP file
response = requests.get(url)
zip_file = ZipFile(BytesIO(response.content))

# Extract the specific CSV file
csv_filename = 'student-mat.csv'
with zip_file.open(csv_filename) as file:
    data = pd.read_csv(file, sep=';')

# Preprocess the data (example steps)
data['final_grade'] = (data['G1'] + data['G2'] + data['G3']) / 3
data['stress_level'] = data['final_grade'].apply(lambda x: 'High' if x < 10 else 'Low')
features = data[['studytime', 'failures', 'absences', 'health']]
target = data['stress_level']

# Encode the target variable
target = target.map({'Low': 0, 'High': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Save the model and scaler
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
