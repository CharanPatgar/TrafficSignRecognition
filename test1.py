from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# Define the classes based on the labels file
labels = pd.read_csv('labels.csv')
class_names = labels['Name'].to_list()

# Path for uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Preprocessing function (same as in the training script)
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255.0
    image = image.reshape(1, 32, 32, 1)  # Add batch dimension and depth
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Read the uploaded image
            image = cv2.imread(file_path)
            image = cv2.resize(image, (32, 32))  # Resize to match training images
            preprocessed_image = preprocess_image(image)

            # Predict the class
            prediction = model.predict(preprocessed_image)
            class_index = np.argmax(prediction)
            class_name = class_names[class_index]

            return render_template('result.html', class_name=class_name, filename=filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
