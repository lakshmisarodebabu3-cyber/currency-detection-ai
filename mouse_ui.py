import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
model = load_model("model.h5")

class_names = ["India", "Thailand", "UAE", "USA"]

# ---------------- GLOBAL VARIABLES ----------------
uploaded_img = None
image_path = None
result_text = ""

# ---------------- CREATE UI ----------------
def draw_ui():
    ui = np.ones((500, 800, 3), dtype=np.uint8) * 255

    # Title
    cv2.putText(ui, "Currency Detection System", (200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Buttons
    cv2.rectangle(ui, (50, 80), (250, 130), (0, 255, 0), -1)
    cv2.putText(ui, "Upload", (110, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.rectangle(ui, (300, 80), (500, 130), (255, 0, 0), -1)
    cv2.putText(ui, "Predict", (355, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.rectangle(ui, (550, 80), (750, 130), (0, 0, 255), -1)
    cv2.putText(ui, "Clear", (620, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Image box
    cv2.rectangle(ui, (50, 160), (350, 360), (0, 0, 0), 2)

    # Result box
    cv2.rectangle(ui, (400, 180), (750, 260), (0, 0, 0), 2)
    cv2.putText(ui, "Result:", (400, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if result_text:
        cv2.putText(ui, result_text, (410, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

    return ui

# ---------------- MOUSE CLICK HANDLER ----------------
def mouse_click(event, x, y, flags, param):
    global uploaded_img, image_path, result_text

    # Upload button
    if event == cv2.EVENT_LBUTTONDOWN and 50 < x < 250 and 80 < y < 130:
        image_path = input("Enter image path: ")
        img = cv2.imread(image_path)

        if img is not None:
            uploaded_img = cv2.resize(img, (300, 200))
            result_text = "Image Loaded"
        else:
            result_text = "Invalid Image Path"

    # Predict button
    elif event == cv2.EVENT_LBUTTONDOWN and 300 < x < 500 and 80 < y < 130:
        if image_path is None:
            result_text = "Upload Image First"
            return

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        index = np.argmax(prediction)
        confidence = prediction[0][index] * 100

        result_text = f"{class_names[index]} ({confidence:.2f}%)"

    # Clear button
    elif event == cv2.EVENT_LBUTTONDOWN and 550 < x < 750 and 80 < y < 130:
        uploaded_img = None
        image_path = None
        result_text = ""

# ---------------- MAIN LOOP ----------------
cv2.namedWindow("Currency Detection")
cv2.setMouseCallback("Currency Detection", mouse_click)

while True:
    ui = draw_ui()

    if uploaded_img is not None:
        ui[160:360, 50:350] = uploaded_img

    cv2.imshow("Currency Detection", ui)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()