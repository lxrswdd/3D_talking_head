# -*- coding: utf-8 -*-

import sys
sys.path.append(r'D:\A\OneDrive - UBC\Projects\3D_HEAD\UI\models')
from FakeModel import FakeModel
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
from PyQt5.QtCore import Qt, QTimer,QRect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QScrollBar
import cv2
import mediapipe as mp
mp_face_detection=mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
mp_drawing=mp.solutions.drawing_utils


from typing import Tuple, Union
import math
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


# model = FakeModel(model_path='./models/fake_model.npy')
class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow,self).__init__()
        self.setGeometry(200,200,600,600) # Location (relative to your monitor) and size of the main window
        self.setWindowTitle("Tutorial 1")
        self.setupUi()
        self.nose_x = 300
        self.yaw = 0
    def setupUi(self):
        # MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(796, 589)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        # Initialize veritcal slider
        self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider.setGeometry(QtCore.QRect(20, 170, 16, 160)) # Location and size of the vertical slider
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")

        # Initialize horizontal slider
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(320, 510, 160, 16)) # Location and size of the horizontal slider
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")

        # Initialize webcam layout
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(200, 200, int(1920/3), int(1080/3))) # Location and size of the webcam layout
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        
        self.image_label = QLabel(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.addWidget(self.image_label)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.setLayout(self.verticalLayout)


        # # Initialize the timer
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.update_frame)

                # Initialize the timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gen)

        # Open the webcam
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            print("Failed to open webcam!")
            sys.exit()

        self.model = FakeModel(model_path=r"D:\A\OneDrive - UBC\Projects\3D_HEAD\UI\models\fake_model.npy")

        # Start the timer
        self.timer.start(300)  # Update every 30 milliseconds

        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 796, 25))
        self.menubar.setObjectName("menubar")

        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")

        self.setStatusBar(self.statusbar)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

    def update_gen(self):
        ret, frame = self.video_capture.read()
        if ret:
    
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results=mp_face_detection.process(frame)

            # draw the face detection annotations on the image
            frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            height, width, _ = frame.shape

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame,detection)
                    keypoints = detection.location_data.relative_keypoints
                    keypoint = keypoints[2]
                    keypoint = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                            width, height)
                    print(keypoint)
        if keypoint[0] > self.nose_x:
            self.yaw +=0.1
        else:
            self.yaw -=0.1
        img =  self.model.render(pitch=-0.15, yaw=self.yaw)
        img = cv2.convertScaleAbs(img, alpha=(255.0))
        
        # Create a QImage from the frame
        image = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(image)

        # Set the QPixmap on the label to display the frame
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.setGeometry(QRect(200, 200, 512, 512))



    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a QImage from the frame
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)

            # Create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(image)

            # Set the QPixmap on the label to display the frame
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.image_label.setGeometry(QRect(200, 200, 512, 512))


if __name__ == "__main__":
    # import sys
    # app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    # ui = Ui_MainWindow()
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    # sys.exit(app.exec_())

    app = QtWidgets.QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(app.exec_())