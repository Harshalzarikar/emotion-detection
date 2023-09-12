import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define the emotions
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# Specify the paths to your train and test datasets
train_data_folder = r"C:\Users\Asus\Desktop\final year project\train"
test_data_folder = r"C:\Users\Asus\Desktop\final year project\test"

X_train = []
y_train = []
X_test = []
y_test = []

# Load images from the train dataset
for emotion_label, emotion in enumerate(emotions):
    train_emotion_folder = os.path.join(train_data_folder, emotion)
    for filename in os.listdir(train_emotion_folder):
        img = Image.open(os.path.join(train_emotion_folder, filename)).convert('L')  # Open image in grayscale mode
        img = img.resize((48, 48))  # Resize to match the model input size
        img_array = np.array(img).flatten()
        X_train.append(img_array)
        y_train.append(emotion_label)

# Load images from the test dataset
for emotion_label, emotion in enumerate(emotions):
    test_emotion_folder = os.path.join(test_data_folder, emotion)
    for filename in os.listdir(test_emotion_folder):
        img = Image.open(os.path.join(test_emotion_folder, filename)).convert('L')  # Open image in grayscale mode
        img = img.resize((48, 48))  # Resize to match the model input size
        img_array = np.array(img).flatten()
        X_test.append(img_array)
        y_test.append(emotion_label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the trained model to a file
joblib.dump(clf, 'emotion_model.pkl')

