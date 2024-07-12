from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import subprocess
import json
import os
import pandas as pd
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import pickle

app = Flask(__name__)

# Dummy user data for login and registration
USER_DATA_FILE = 'users.json'

def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f)

users = load_users()

# Load class names from CSV
class_names = pd.read_csv('labels.csv', index_col=0).iloc[:, 0].to_dict()
images_path = 'images'  
audio_path = 'static/audio'  

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']
    users = load_users()
    if username in users and users[username] == password:
        return redirect(url_for('dashboard'))
    else:
        error_message = 'Invalid credentials. Please try again.'
        return render_template('login.html', error=error_message)
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users:
            error_message = 'Username already exists. Please try another one.'
            return render_template('register.html', error=error_message)
        else:
            users[username] = password
            save_users(users)
            # Reload users to ensure the latest data is available
            users = load_users()
            return redirect(url_for('login'))
    return render_template('register.html', error=None)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
    

@app.route('/logout')
def logout():
    response = redirect(url_for('login'))
    response.headers['Cache-Control'] = 'no-store'
    return response

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
