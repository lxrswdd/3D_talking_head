# -*- coding: utf-8 -*-
import os
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

model_path=r"D:\A\OneDrive - UBC\Projects\3D_HEAD\UI\models\fake_model.npy"

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow,self).__init__()
        self.setStyleSheet("background-color: white;")
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        self.screen_h = sizeObject.height()
        self.screen_w = sizeObject.width()
        print(" Screen size : "  + str(self.screen_h) + "x"  + str(self.screen_w))

        self.setGeometry(int(sizeObject.width()/3),int(sizeObject.height()/3),int(sizeObject.width()/2),int(sizeObject.height()/2)) # Location (relative to your monitor) and size of the main window
        self.setWindowTitle("Tutorial 1")
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.setupUi()
        self.video_capture = cv2.VideoCapture(1)

        ret, frame = self.video_capture.read()

        if ret:
    
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results=mp_face_detection.process(frame)
            # draw the face detection annotations on the image
            height, width, _ = frame.shape

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame,detection)
                    keypoints = detection.location_data.relative_keypoints
                    keypoint = keypoints[2]
                    keypoint = (keypoint.x, keypoint.y)

                    print('Inital face coordinations: ',keypoint)
            else:
                print("Failed to detect a face.")
        else:
            print("Failed to read webcam.")

        self.nose_x = keypoint[0] # The inital nose x coordinate
        self.nose_y = keypoint[1] # The inital nose y coordinate
        self.delta_x_prev = 0
        self.delta_y_prev = 0
        
        self.yaw = 0
        self.pitch = 0


    def setupUi(self):
        # MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(796, 589)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        img_size = self.screen_h - 50
        img_pos_x = (self.screen_w - img_size) // 2
        img_pos_y = 25

        # Initialize webcam layout
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(img_pos_x, img_pos_y, img_size, img_size)) # Location and size of the webcam layout
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        
        self.image_label = QLabel(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.addWidget(self.image_label)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.setLayout(self.verticalLayout)


        # Initialize the timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gen)

        # Open the webcam
        self.video_capture = cv2.VideoCapture(1)
        if not self.video_capture.isOpened():
            print("Failed to open webcam!")
            sys.exit()
        print()
        self.model = FakeModel(model_path=r"D:\OneDrive - UBC\Projects\3D_HEAD\UI\models\fake_model.npy")

        # Start the timer
        self.timer.start(100)  # Update every 30 milliseconds

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
                detection = results.detections[0]

                keypoints = detection.location_data.relative_keypoints
                keypoint = keypoints[2]
                keypoint = (keypoint.x, keypoint.y)

                # compute the x,y offset
                delta_x = self.nose_x - keypoint[0]
                delta_y = self.nose_y - keypoint[1]

                # smoothing the x,y offset
                gamma = 0.5
                delta_x_smoothed = (1-gamma)*self.delta_x_prev + gamma*delta_x
                delta_y_smoothed = (1-gamma)*self.delta_y_prev + gamma*delta_y
                self.delta_x_prev = delta_x
                self.delta_y_prev = delta_y


                self.yaw = -1.5*delta_x
                if self.yaw >= 0.45:
                    self.yaw = 0.45

                self.pitch = 0.2*delta_y
                if self.pitch >= 0.15:
                    self.pitch = 0.15



            else:
                print('Face not detected')
        
        img =  self.model.render(pitch=self.pitch, yaw=self.yaw)
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

    app = QtWidgets.QApplication(sys.argv)
    win = Ui_MainWindow()
    print('debug1')
    win.show()
    print('debug2')

    sys.exit(app.exec_())


# EOF