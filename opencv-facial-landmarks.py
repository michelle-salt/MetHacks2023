import cv2

# Load the Haar Cascades face detector and facial landmark detection model
face_cascade = cv2.CascadeClassifier('opencv-files\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv-files\haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('opencv-files\haarcascade_smile.xml')

# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through each detected face
    for (x,y,w,h) in faces:
        # Extract the region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

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

    # Show the resulting video stream
    cv2.imshow('Facial Landmark Detection', frame)

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
