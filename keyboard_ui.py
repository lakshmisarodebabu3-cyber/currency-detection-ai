import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 🔥 Hides ALL TF logs
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog
import tkinter as tk
import os

# ---------------- LOAD MODEL ----------------
model = load_model("model.h5")

# ⚠️ MUST MATCH TRAINING FOLDER ORDER
class_names = ["India", "Thailand", "UAE", "USA"]

# ---------------- TK FOR FILE DIALOG ONLY ----------------
root = tk.Tk()
root.withdraw()

# ---------------- GLOBALS ----------------
ui = np.zeros((500, 700, 3), dtype=np.uint8)
image_path = None
currency_img = None

# ---------------- DRAW UI ----------------
def draw_ui():
    ui[:] = (30, 30, 30)

    cv2.putText(ui, "Currency Detection System",
                (160, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    # Upload button
    cv2.rectangle(ui, (50, 100), (300, 150), (255, 0, 0), 2)
    cv2.putText(ui, "U - Upload Image",
                (70, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    # Predict button
    cv2.rectangle(ui, (50, 180), (300, 230), (0, 255, 0), 2)
    cv2.putText(ui, "P - Predict Currency",
                (60, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
                

    # Quit button
    cv2.rectangle(ui, (50, 260), (300, 310), (0, 0, 255), 2)
    cv2.putText(ui, "Q - Quit",
                (140, 295),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

# ---------------- MAIN LOOP ----------------
draw_ui()

while True:
    cv2.imshow("Currency Detection UI (OpenCV)", ui)
    key = cv2.waitKey(1) & 0xFF

    # -------- UPLOAD IMAGE --------
    if key == ord('u'):
        image_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )

        if image_path and os.path.exists(image_path):
            currency_img = cv2.imread(image_path)

            if currency_img is not None:
                currency_img = cv2.resize(currency_img, (300, 200))
                draw_ui()
                ui[120:320, 350:650] = currency_img

    # -------- PREDICT --------
    elif key == ord('p') and image_path:
        test_img = cv2.imread(image_path)

        if test_img is None:
            continue

        test_img = cv2.resize(test_img, (224, 224))
        test_img = test_img / 255.0
        test_img = np.expand_dims(test_img, axis=0)

        # 🔥 Silence TensorFlow logs
        pred = model.predict(test_img, verbose=0)
        index = np.argmax(pred)
        confidence = pred[0][index] * 100

        # Display result (DO NOT redraw full UI)
        cv2.putText(ui,
                    f"Detected: {class_names[index]}",
                    (350, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(ui,
                    f"Confidence: {confidence:.2f}%",
                    (350, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    # -------- QUIT --------
    elif key == ord('q'):
        break

cv2.destroyAllWindows()