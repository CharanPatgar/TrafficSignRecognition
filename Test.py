import numpy as np
import cv2
import pickle
from flask import Flask

app = Flask(__name__)

#############################################

frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

##############################################

# IMPORT THE TRAINED MODEL
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    # Your class names here...
    return "Class " + str(classNo)

# Function to start camera and prediction loop
def start_detection():
    # SETUP THE VIDEO CAMERA
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)

    while True:
        # READ IMAGE
        success, imgOrignal = cap.read()

        # PROCESS IMAGE
        img = np.asarray(imgOrignal)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        cv2.imshow("Processed Image", img)
        img = img.reshape(1, 32, 32, 1)
        cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        # PREDICT IMAGE
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)
        if probabilityValue > threshold:
            cv2.putText(imgOrignal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue*100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to start camera and detection loop
start_detection()
