# === preprocessing_dataset.py ===
import cv2
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flask import Flask, request, render_template

# Load dataset
def load_dataset(folder_path, size=(100, 100)):
    data, labels = [], []
    classes = os.listdir(folder_path)

    for class_label in classes:
        class_folder = os.path.join(folder_path, class_label)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            features = img.flatten()
            data.append(features)
            labels.append(class_label)

    return np.array(data), np.array(labels)

# Path ke folder dataset
X, y = load_dataset('dataset')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluasi
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Simpan model
joblib.dump(clf, 'model.pkl')

# === Flask Web App ===
app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100)).flatten().reshape(1, -1)
            prediction = model.predict(img)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
