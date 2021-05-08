from imutils.object_detection import non_max_suppression 
import numpy as np 
import imutils 
import cv2 
import time 
import argparse 

# Opencv pre-trained SVM with HOG people features 
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
DEFAULT_WIDTH = 800

def detector(image):
  '''
  @image is a numpy array
  '''

  image = imutils.resize(image, width=min(DEFAULT_WIDTH, image.shape[1]))

  (rects, weights) = HOGCV.detectMultiScale(image, winStride=(8, 8), padding=(4, 4), scale=1.01)

  # Applies non-max supression from imutils package to kick-off overlapped
  # boxes
  rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
  result = non_max_suppression(rects, probs=None, overlapThresh=0.65)

  return result

def localDetect(image_path):
  result = []
  image = cv2.imread(image_path)
  if len(image) <= 0:
      print("[ERROR] could not read your local image")
      return result
  print("[INFO] Detecting people")
  result = detector(image)

  # shows the result
  for (xA, yA, xB, yB) in result:
      cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

  cv2.imshow("result", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  return (result, image)

def cameraDetect():
  cap = cv2.VideoCapture(0)

  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=min(DEFAULT_WIDTH, frame.shape[1]))
    result = detector(frame.copy())

    # shows the result
    for (xA, yA, xB, yB) in result:
      cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()

def convert_to_base64(image):
  image = imutils.resize(image, width=DEFAULT_WIDTH)
  img_str = cv2.imencode('.png', image)[1].tostring()
  b64 = base64.b64encode(img_str)
  return b64.decode('utf-8')

def detectPeople(args):
  image_path = args["image"]
  camera = True if str(args["camera"]) == 'true' else False

  # Routine to read local image
  if image_path != None and not camera:
    print("[INFO] Image path provided, attempting to read image")
    (result, image) = localDetect(image_path)

  # Routine to read images from webcam
  if camera:
    print("[INFO] reading camera images")
    cameraDetect()

def buildPayload(variable, value, context):
  return {variable: {"value": value, "context": context}}

def argsParser():
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", default=None, help="path to image test file directory")
  ap.add_argument("-c", "--camera", default=False, help="Set as true if you wish to use the camera")
  args = vars(ap.parse_args())

  return args

def main():
  args = argsParser()
  detectPeople(args)


if __name__ == '__main__':
  main()