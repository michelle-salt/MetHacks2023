import cv2
import numpy as np
from keras.models import load_model
from time import sleep

# Load the pre-trained CNN model
model = load_model('fer2013_mini_XCEPTION.119-0.65.hdf5', compile=False)

# Load the haarcascade frontal face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to recognize emotions from a video stream
def detect_emotions():

    # Open the device's camera
    cap = cv2.VideoCapture(0)

    # Loop over frames from the camera
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # For each detected face, predict the emotion and draw a rectangle around the face
        for (x, y, w, h) in faces:
            # Extract the face ROI and resize it to 48x48 pixels
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))

            # Normalize the ROI
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=-1)

            # Make a prediction on the ROI, then look up the class name
            preds = model.predict(np.expand_dims(roi, axis=0))[0]
            label = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"][np.argmax(preds)]

            # Draw a rectangle around the face and label it with the predicted emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the frame with the labeled faces
        cv2.imshow('Facial Emotion Detection', frame)

        # Wait for a key press and check if the 'q' key was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to detect emotions from the video stream
detect_emotions()
