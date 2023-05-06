import json
import requests
from PIL import Image

# Load the API key and endpoint URL from config.json
with open('config.json', 'r') as f:
    config = json.load(f)
api_key = config['api_key']
endpoint_url = config['endpoint_url']

# Define the Face API endpoint URL and headers
detect_url = endpoint_url + '/detect'
headers = {
    'Ocp-Apim-Subscription-Key': api_key,
    'Content-Type': 'application/octet-stream'
}

# Load an image file
try:
    with open('images\white-man.jpg', 'rb') as f:
        image_file = f.read()
except FileNotFoundError:
    print('Error: Could not find image file')
    exit()

# Call the Face API to detect faces in the image
response = requests.post(detect_url, headers=headers, data=image_file)

# Check for errors in the API response
if response.status_code != 200:
    print(f'Error: {response.status_code} {response.json()["error"]["message"]}')
    exit()

# Parse the response JSON to get the face rectangles for each face detected
results = json.loads(response.content)
for result in results:
    face_rect = result['faceRectangle']
    print(f'Face Rectangle: {face_rect}')

    # Crop the image to the detected face
    image = Image.open('images\white-man.jpg')
    cropped_image = image.crop((face_rect['left'], face_rect['top'], face_rect['left'] + face_rect['width'], face_rect['top'] + face_rect['height']))
    cropped_image.save('cropped_image.jpg')

    # Call the Face API to detect emotions in the face
    detect_url = endpoint_url + '/detect?returnFaceAttributes=emotion'
    with open('cropped_image.jpg', 'rb') as f:
        image_file = f.read()
    response = requests.post(detect_url, headers=headers, data=image_file)

    # Check for errors in the API response
    if response.status_code != 200:
        print(f'Error: {response.status_code} {response.json()["error"]["message"]}')
        exit()

    # Parse the response JSON to get the emotions detected in the face
    face_results = json.loads(response.content)
    for face_result in face_results:
        emotions = face_result['faceAttributes']['emotion']
        print(f'Emotions: {emotions}')
