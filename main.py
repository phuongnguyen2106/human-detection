import sys

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QFileDialog, QPushButton, QLabel, QRadioButton
from PyQt5.QtGui import QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, Qt, QThread

import cv2
import imutils
import numpy as np
import hogs
import haar

class VideoThread(QThread):
  change_pixmap_signal = pyqtSignal(np.ndarray)

  def __init__(self, parent=None):
    """Initializer."""
    super().__init__(parent)
    self.path = None
    self.cap = None
    self.pause = True
  
  def setPath(self, path):
    self.path = path

  def run(self):
    if self.path is not None:
      self.cap = cv2.VideoCapture(self.path)
    else:
      # capture from web cam
      self.cap = cv2.VideoCapture(0)

    while self.cap.isOpened():
      self.pause = True
      ret, cv_img = self.cap.read()
      if ret:
        self.change_pixmap_signal.emit(cv_img)
        while self.pause:
          pass        

  def stop(self):
    if self.cap is not None:
      print("release cap")
      self.cap.release()
    # self.quit()
    # self.wait()
class App(QWidget):
  """Main Window."""
  def __init__(self, parent=None):
    """Initializer."""
    super().__init__(parent)
    self.title = 'Human detection app'
    self.left = 10
    self.top = 10
    self.width = 900
    self.height = 800
    
    self.image = QLabel("Hello")
    self.imageWidth = 800
    self.imageHeight = 600

    self.method = 'HOGs'
    self.thread = None

    self._initWindow()
    self._initLayout()
    self._initDefaultImage()

  def _initWindow(self):
    self.setWindowTitle(self.title)
    self.setGeometry(self.left, self.top, self.width, self.height)

  def _initLayout(self):
    layout = QGridLayout()

    label = QLabel("Chọn phương pháp :")
    layout.addWidget(label, 0, 0)

    b1 = QRadioButton("SIFT")    
    b1.toggled.connect(lambda:self.btnstate(b1))
    layout.addWidget(b1, 0, 1)

    b2 = QRadioButton("ORB")
    b2.toggled.connect(lambda:self.btnstate(b2))
    layout.addWidget(b2, 0, 2)

    b3 = QRadioButton("HOGs")
    b3.setChecked(True)
    b3.toggled.connect(lambda:self.btnstate(b3))
    layout.addWidget(b3, 0, 3)

    b4 = QRadioButton("HAAR")
    b4.toggled.connect(lambda:self.btnstate(b4))
    layout.addWidget(b4, 0, 4)

    b5 = QPushButton(text="Chọn ảnh")
    b5.clicked.connect(self.getImage)
    layout.addWidget(b5, 1, 0)

    b6 = QPushButton(text="Chọn video")
    b6.clicked.connect(self.getVideo)
    layout.addWidget(b6, 1, 2)

    b7 = QPushButton(text="Kết nối webcam")
    b7.clicked.connect(self.connectWebcam)
    layout.addWidget(b7, 1, 4)
    
    layout.addWidget(self.image, 2, 0, 5, 10)

    self.setLayout(layout)

  def _initDefaultImage(self):
    grey = QPixmap(self.imageWidth, self.imageHeight)
    grey.fill(QColor('darkGray'))
    self.image.setPixmap(grey)

  def _closeThread(self):
    if self.thread is not None and self.thread.isRunning():
      print("thread running")
      self.thread.stop()
      print("thread stop")
      
      del self.thread
      self.thread = None

  def btnstate(self,b):
    if b.isChecked() == True:
      self.method = b.text()
      print(self.method)
	
  def getImage(self):
    self._closeThread()
    self._initDefaultImage()

    fname = QFileDialog.getOpenFileName(self, 'Open file', "./", "Image files (*.jpg *.gif *.png)")
    self.imagePath = fname[0]    
    self.imageProcess()

  def getVideo(self):
    self._closeThread()
    self._initDefaultImage()

    fname = QFileDialog.getOpenFileName(self, 'Open file', "./", "Video files (*.mp4)")
    self.imagePath = fname[0]
    if self.imagePath is not None or self.imagePath.length() > 0:
      self.videoProcess()

  def connectWebcam(self):
    self._closeThread()
    self._initDefaultImage()

    # create the video capture thread
    self.thread = VideoThread()
    # connect its signal to the update_image slot
    self.thread.change_pixmap_signal.connect(self.detectImage)
    # start the thread
    self.thread.start()

  def imageProcess(self):
    if self.method == 'HOGs':
      image = hogs.imageDetect(self.imagePath)
    elif self.method == 'HAAR':
      image = haar.imageDetect(self.imagePath)
      
    self.updateImage(image)

  def videoProcess(self):
    # create the video capture thread
    self.thread = VideoThread()
    self.thread.setPath(self.imagePath)
    # connect its signal to the update_image slot
    self.thread.change_pixmap_signal.connect(self.detectImage)
    # start the thread
    self.thread.start()

  @pyqtSlot(np.ndarray)
  def detectImage(self, cv_img):
    if self.method == 'HOGs':
      image = hogs.imageDetect('', cv_img)
    elif self.method == 'HAAR':
      image = haar.imageDetect('', cv_img)

    qt_img = self.convertCV2Qt(image)
    self.image.setPixmap(qt_img)
    if self.thread is not None:
      self.thread.pause = False

  def updateImage(self, cv_img):
    """Updates the image_label with a new opencv image"""
    qt_img = self.convertCV2Qt(cv_img)
    self.image.setPixmap(qt_img)

  def convertCV2Qt(self, cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(self.imageWidth, self.imageHeight, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)


if __name__ == '__main__':
  app = QApplication(sys.argv)
  win = App()
  win.show()
  sys.exit(app.exec_())