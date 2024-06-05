import numpy as np
import cv2
import os
import pickle
import pandas as pd
from werkzeug.utils import secure_filename

# Load the trained model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# Define the classes based on the labels file
labels = pd.read_csv('labels.csv')
class_names = labels['Name'].to_list()

# Path for uploaded images
UPLOAD_FOLDER = 'uploads'

# Preprocessing function 
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255.0
    image = image.reshape(1, 32, 32, 1) 
    return image

def process_uploaded_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32))  
    preprocessed_image = preprocess_image(image)

    # Predict the class
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]

    return class_name
