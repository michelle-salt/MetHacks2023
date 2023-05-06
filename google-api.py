import io
import os


# Set the path to your service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'methacks2023-385904-e4c5d5761749.json'

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision_v1 import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to analyze
file_name = os.path.abspath('images\white-man.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()
image = types.Image(content=content)

# Detects emotions in the image
response = client.face_detection(image=image)
faces = response.face_annotations
for face in faces:
    likelihood = face.likelihood
    emotions = ['joy', 'sorrow', 'anger', 'surprise']
    print('Emotions:')
    for emotion in emotions:
        print(f'{emotion}: {getattr(likelihood, emotion)}')
