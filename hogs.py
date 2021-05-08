from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2

# Opencv pre-trained SVM with HOG people features 
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
DEFAULT_WIDTH = 800

def detector(image):
  '''
  @image is a numpy array
  '''

  image = imutils.resize(image, width=min(DEFAULT_WIDTH, image.shape[1]))
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  (rects, weights) = HOGCV.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

  # Applies non-max supression from imutils package to kick-off overlapped
  # boxes
  rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
  result = non_max_suppression(rects, probs=None, overlapThresh=0.65)

  return result

def imageDetect(imagePath, image=None):
  result = []
  if image is None:
    image = cv2.imread(imagePath)
  if len(image) <= 0:
    print("[ERROR] could not read your local image")
    return result
  print("[INFO] Detecting people")
  result = detector(image)

  # shows the result
  for (xA, yA, xB, yB) in result:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

  return image