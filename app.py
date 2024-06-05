from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import subprocess
import os
import pandas as pd
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import pickle

app = Flask(__name__)

# Dummy user data for login
users = {
    "user1": "password1",
    "user2": "password2"
}

# Load class names from CSV
class_names = pd.read_csv('labels.csv', index_col=0).iloc[:, 0].to_dict()
images_path = 'images'  
audio_path = 'static/audio'  

# Mock data for class descriptions and audio files
class_descriptions = {
    'stop': 'This is a stop sign. It means you must come to a complete stop.',
    'yield': 'This is a yield sign. It means you must yield to other traffic.',
    
}
audio_files = {
    'stop': 'stop.mp3',
    'yield': 'yield.mp3',
    
}

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']

    if username in users and users[username] == password:
        return redirect(url_for('dashboard'))
    else:
        return "Invalid credentials. Please try again."

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/run-test')
def run_test():
    subprocess.Popen(["python", "test.py"])  
    return 'Test.py executed successfully!'

@app.route('/traffic-signs')
def traffic_signs():
    images_path = os.path.join(app.static_folder, 'images')
    images = [img for img in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, img))]
    traffic_signs = [{'name': os.path.splitext(img)[0], 'image': f'images/{img}'} for img in images]
    return render_template('traffic_signs.html', traffic_signs=traffic_signs)

@app.route('/upload-images', methods=['GET', 'POST'])
def upload_images():
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

            # Call the function from test1.py
            class_name = process_uploaded_image(file_path)
            return render_template('result.html', class_name=class_name, filename=filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_uploaded_image(file_path):
    with open("model_trained.p", "rb") as f:
        model = pickle.load(f)

    labels = pd.read_csv('labels.csv')
    class_names = labels['Name'].to_list()

    def preprocess_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        image = image / 255.0
        image = image.reshape(1, 32, 32, 1)
        return image

    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32))
    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]

    return class_name

if __name__ == '__main__':
    app.run(debug=True)
