import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained CNN model for facial expression recognition
model = load_model('fer2013_mini_XCEPTION.119-0.65.hdf5')

# Define the emotions that the model can recognize
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the haarcascade frontal face classifier
face_cascade = cv2.CascadeClassifier('opencv-files\haarcascade_frontalface_default.xml')

# Load the image and convert it to grayscale
img = cv2.imread('images\happy-woman-with-sad-man.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# For each detected face, predict the emotion and draw a rectangle around it
for (x, y, w, h) in faces:
    # Extract the face ROI
    face_roi = gray[y:y+h, x:x+w]
    # Resize the face ROI to 48x48 pixels (the input size of the CNN model)
    face_roi = cv2.resize(face_roi, (48, 48))
    # Normalize the face ROI pixels between 0 and 1
    face_roi = face_roi / 255.0
    # Reshape the face ROI to a 4D tensor with shape (1, 48, 48, 1)
    face_roi = np.reshape(face_roi, (1, 48, 48, 1))
    # Predict the emotion using the CNN model
    emotion_probabilities = model.predict(face_roi)
    # Get the index of the highest probability
    predicted_emotion_index = np.argmax(emotion_probabilities)
    # Get the predicted emotion label
    predicted_emotion = emotions[predicted_emotion_index]
    # Draw a rectangle around the face and label it with the predicted emotion
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, predicted_emotion, (x+5, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with the detected faces and their predicted emotions
cv2.imshow('Facial Emotion Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
