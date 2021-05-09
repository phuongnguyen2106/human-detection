import cv2
import numpy as np

# Create our body classifier
body_classifier = cv2.CascadeClassifier('./resources/haarcascade_frontalface_default.xml')

def detector(image):
  '''
  @image is a numpy array
  '''

  image = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Pass frame to our body classifier
  bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

  # Extract bounding boxes for any bodies identified
  for (x,y,w,h) in bodies:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

  return image

def imageDetect(imagePath, image=None):
  if image is None:
    image = cv2.imread(imagePath)
  if len(image) <= 0:
    print("[ERROR] could not read your local image")
    return image
  print("[INFO] Detecting people")
  return detector(image)