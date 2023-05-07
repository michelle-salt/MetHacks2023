import cv2
import numpy as np
from keras.models import load_model
from time import sleep

# Load the pre-trained CNN model
model = load_model('fer2013_mini_XCEPTION.119-0.65.hdf5', compile=False)

# Load the haarcascade frontal face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the Haar Cascades face detector and facial landmark detection model
face_cascade = cv2.CascadeClassifier('opencv-files\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv-files\haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('opencv-files\haarcascade_smile.xml')

# Open the device's camera
cap = cv2.VideoCapture(0)

# Define a function to recognize emotions from a video stream
def detect_emotions():

    # Loop over frames from the camera
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # # Detect faces in the grayscale image
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Loop through each detected face, and label landmarks + emotions
        for (x,y,w,h) in faces:
            # Extract the region of interest (ROI) for the face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

             # Extract the face ROI and resize it to 48x48 pixels
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            
            # Normalize the ROI
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=-1)

            # Make a prediction on the ROI, then look up the class name
            preds = model.predict(np.expand_dims(roi, axis=0))[0]
            label = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"][np.argmax(preds)]

            # Detect eyes in the face ROI
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Loop through each detected eye and draw a dot
            for (ex,ey,ew,eh) in eyes:
                cv2.circle(roi_color, (ex+int(ew/2),ey+int(eh/2)), 5, (0,255,0), -1)

            # Detect mouth in the face ROI
            mouths = mouth_cascade.detectMultiScale(roi_gray)

            # Loop through each detected mouth and draw a dot
            for (mx,my,mw,mh) in mouths:
                cv2.circle(roi_color, (mx+int(mw/2),my+int(mh/2)), 5, (0,0,255), -1)

            # Draw a rectangle around the face and label it with the predicted emotion
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the resulting video stream
        cv2.imshow('Facial Analysis', frame)

        # Exit the program when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break       

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to detect emotions from the video stream
detect_emotions()