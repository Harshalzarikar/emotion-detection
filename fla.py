import os
import numpy as np
from PIL import Image
import joblib
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
clf = joblib.load('emotion_model.pkl')

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Specify the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Load the uploaded image, preprocess it, and make a prediction
        img = Image.open(filepath).convert('L')  # Convert image to grayscale
        img = img.resize((48, 48))  # Resize to match the model input size
        img_array = np.array(img).flatten()
        emotion_label = clf.predict([img_array])[0]
        detected_emotion = emotions[emotion_label]

        # Delete the uploaded image after processing
        os.remove(filepath)

        return render_template('result.html', emotion=detected_emotion)

    else:
        return render_template('index.html', message='Invalid file extension')

if __name__ == '__main__':
    app.run(debug=True)
