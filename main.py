import sys

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QFileDialog, QPushButton, QLabel, QRadioButton
from PyQt5.QtGui import QPixmap, QColor, QImage
from PyQt5.QtCore import Qt

import cv2
import hogs


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
    self.method = 'HOGs'

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
    layout.addWidget(b6, 1, 2)

    b7 = QPushButton(text="Kết nối webcam")
    layout.addWidget(b7, 1, 4)

    self.image = QLabel("Hello")
    layout.addWidget(self.image, 2, 0, 5, 10)

    self.setLayout(layout)

  def _initDefaultImage(self):
    self.imageWidth = 800
    self.imageHeight = 600

    # self.imagePath = './fpt_uni.png'
    grey = QPixmap(self.imageWidth, self.imageHeight)
    grey.fill(QColor('darkGray'))
    self.image.setPixmap(grey)

  def btnstate(self,b):
    if b.isChecked() == True:
      self.method = b.text()
      print(self.method)
	
  def getImage(self):
    fname = QFileDialog.getOpenFileName(self, 'Open file', "./", "Image files (*.jpg *.gif *.png)")
    self.imagePath = fname[0]    
    self.imageProcess()

  def imageProcess(self):
    if self.method == 'HOGs':
      image = hogs.imageDetect(self.imagePath)
      pixmap = self.convert_cv_qt(image)
      self.image.setPixmap(QPixmap(pixmap))
      self.resize(pixmap.width(), pixmap.height())

  def convert_cv_qt(self, cv_img):
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