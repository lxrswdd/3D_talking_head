import sys
import cv2
from PyQt5.QtCore import Qt, QTimer,QRect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QScrollBar

class WebcamWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Create a label to display the webcam feed
        self.image_label = QLabel(self)

        # Set up the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        self.image_label.setGeometry(QRect(200, 200, 512, 512))

        self.setLayout(layout)

        # Create a timer to update the webcam feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Open the webcam
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            print("Failed to open webcam!")
            sys.exit()

        # Start the timer
        self.timer.start(30)  # Update every 30 milliseconds

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


# Create the application
app = QApplication(sys.argv)

# Create the main window
window = QMainWindow()
window.setWindowTitle('Webcam Viewer')
window.setGeometry(100, 100, 900, 900)

# Create a central widget for the main window
central_widget = QWidget(window)
window.setCentralWidget(central_widget)

# Create a QVBoxLayout to hold the webcam widget
layout = QVBoxLayout(central_widget)

# Create the webcam widget and add it to the layout
webcam_widget = WebcamWidget()
webcam_widget.setGeometry(QRect(200, 200, 512, 512))
layout.addWidget(webcam_widget)

# Create vertical and horizontal sliders
vertical_slider = QScrollBar(Qt.Vertical, central_widget)
horizontal_slider = QScrollBar(Qt.Horizontal, central_widget)

# Add the sliders to the layoutg
layout.addWidget(vertical_slider)
layout.addWidget(horizontal_slider)

# Show the main window
window.show()

# Run the application's event loop
sys.exit(app.exec_())
