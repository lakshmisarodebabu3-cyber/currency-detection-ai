import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model("model.h5")
class_names = ['India', 'Thailand', 'UAE', 'USA']

image_path = None
for ext in ['jpg', 'jpeg', 'png']:
    if os.path.exists(f"test.{ext}"):
        image_path = f"test.{ext}"
        break

if image_path is None:
    print("❌ test image not found")
    exit()

img = cv2.imread(image_path)
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
print("💰 Currency detected:", class_names[np.argmax(prediction)])