from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
import pandas as pd

app = Flask(__name__)

# Dummy user data (replace this with your actual user authentication)
users = {
    "user1": "password1",
    "user2": "password2"
}

# Load class names from CSV
class_names = pd.read_csv('labels.csv', index_col=0).iloc[:, 0].to_dict()
images_path = 'images'  # Directory where traffic sign images are stored

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
    subprocess.Popen(["python", "test.py"])  # Change the path as needed
    return 'Test.py executed successfully!'

@app.route('/traffic-signs')
def traffic_signs():
    images_path = os.path.join(app.static_folder, 'images')
    images = [img for img in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, img))]
    traffic_signs = [{'name': os.path.splitext(img)[0], 'image': f'images/{img}'} for img in images]
    return render_template('traffic_signs.html', traffic_signs=traffic_signs)

if __name__ == '__main__':
    app.run(debug=True)
