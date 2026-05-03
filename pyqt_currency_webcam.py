import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from tensorflow.keras.models import load_model
import datetime

# ---------------- LOAD MODEL ----------------
model = load_model("model.h5")
class_names = ["India", "Thailand", "UAE", "USA"]

# ---------------- MAIN UI CLASS ----------------
class CurrencyDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Currency Detection System")
        self.setGeometry(100, 100, 700, 650)

        # UI elements
        self.image_label = QLabel(self)
        self.pred_label = QLabel("Prediction: ", self)
        self.save_btn = QPushButton("Save Screenshot", self)
        self.save_btn.clicked.connect(self.save_screenshot)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.pred_label)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

        # Start webcam
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS backend
        self.frame = None
        self.pred_text = ""

        # Timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms ~ 33 FPS

    def update_frame(self):
        ret, temp = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(temp, (640, 480))
        self.frame = frame.copy()

        # Prediction
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)
        idx = np.argmax(pred)
        conf = pred[0][idx] * 100
        self.pred_text = f"{class_names[idx]} ({conf:.1f}%)"
        self.pred_label.setText("Prediction: " + self.pred_text)

        # Draw bounding box on RGB frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(rgb_frame, (100, 50), (540, 430), (0, 255, 0), 2)
        cv2.putText(rgb_frame, self.pred_text, (110, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show image in QLabel
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def save_screenshot(self):
        if self.frame is None:
            return
        # Draw prediction on the saved image
        save_frame = self.frame.copy()
        cv2.rectangle(save_frame, (100, 50), (540, 430), (0, 255, 0), 2)
        cv2.putText(save_frame, self.pred_text, (110, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"currency_{timestamp}.jpg"
        cv2.imwrite(filename, save_frame)
        print(f"Saved screenshot as {filename}")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    detector = CurrencyDetector()
    detector.show()
    sys.exit(app.exec_())