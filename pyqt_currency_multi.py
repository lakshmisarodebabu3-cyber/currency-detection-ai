import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from tensorflow.keras.models import load_model
import datetime

# Load your trained model
model = load_model("model.h5")
class_names = ["India", "Thailand", "UAE", "USA"]

class MultiCurrencyDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Currency Detection")
        self.setGeometry(100, 100, 700, 650)

        self.image_label = QLabel(self)
        self.pred_label = QLabel("Predictions: ", self)
        self.save_btn = QPushButton("Save Screenshot", self)
        self.save_btn.clicked.connect(self.save_screenshot)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.pred_label)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

        # Webcam
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        self.frame = None
        self.pred_texts = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, (640, 480))
        self.frame = frame.copy()
        display_frame = frame.copy()
        self.pred_texts = []

        # Convert to gray + threshold for contour detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 50:
                continue  # ignore too small regions
            roi = frame[y:y+h, x:x+w]
            try:
                img = cv2.resize(roi, (224,224))
            except:
                continue
            img = img/255.0
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img, verbose=0)
            idx = np.argmax(pred)
            conf = pred[0][idx]*100
            label = f"{class_names[idx]} {conf:.1f}%"
            self.pred_texts.append(label)
            # Draw box + label
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

        # Update QLabel
        self.pred_label.setText("Predictions: " + ", ".join(self.pred_texts))
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch*w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def save_screenshot(self):
        if self.frame is None:
            return
        save_frame = self.frame.copy()
        for cnt in cv2.findContours(cv2.cvtColor(save_frame, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 50:
                continue
            roi = save_frame[y:y+h, x:x+w]
            try:
                img = cv2.resize(roi, (224,224))
            except:
                continue
            img = img/255.0
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img, verbose=0)
            idx = np.argmax(pred)
            conf = pred[0][idx]*100
            label = f"{class_names[idx]} {conf:.1f}%"
            cv2.rectangle(save_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(save_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"currency_multi_{timestamp}.jpg"
        cv2.imwrite(filename, save_frame)
        print(f"Saved screenshot as {filename}")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__=="__main__":
    app = QApplication(sys.argv)
    detector = MultiCurrencyDetector()
    detector.show()
    sys.exit(app.exec_())