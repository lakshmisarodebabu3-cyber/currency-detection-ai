💱 Currency Detection AI

📌 Project Overview

This project is a Currency Detection System using Artificial Intelligence (CNN). It identifies currency notes from multiple countries such as India, Thailand, UAE, and USA using image classification.

The system supports multiple input methods including:

- Mouse-based image upload
- Keyboard-controlled prediction
- Real-time webcam detection

---

🚀 Features

- Multi-currency detection (4 countries)
- Real-time webcam prediction
- Keyboard-based detection
- Mouse-based image upload
- CNN-based classification
- Displays prediction with confidence score

---

🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow

---

📂 Project Structure

currency_detection_ai/
│
├── train.py
├── predict.py
├── keyboard.py
├── mouse.py
├── currency_webcam.py
├── currency_multi.py
├── opencv_ui.py
├── launcher.py
├── README.md

---

⚙️ Setup Instructions

1️⃣ Clone the Repository

git clone https://github.com/your-username/currency-detection-ai.git
cd currency-detection-ai

---

2️⃣ Create Virtual Environment (Optional)

python -m venv venv
source venv/bin/activate   (Mac/Linux)
venv\Scripts\activate      (Windows)

---

3️⃣ Install Dependencies

pip install tensorflow opencv-python numpy pillow

---

▶️ How to Run the Project

🔹 Train the Model

python train.py

---

🔹 Predict from Image

python predict.py

---

🔹 Keyboard Mode

python keyboard.py
Press P to predict

---

🔹 Mouse Mode

python mouse.py
Select image using mouse

---

🔹 Webcam Detection

python currency_webcam.py
Press W to start webcam
Press Q to quit

---

🔹 Multi-Currency Detection

python currency_multi.py

---

🔹 Run Full UI

python opencv_ui.py

---

🔹 Run Launcher (Main File)

python launcher.py

---

📊 Algorithm Used

- Convolutional Neural Network (CNN)
- Image preprocessing (resize, normalization)
- Feature extraction using Conv2D layers
- Classification using Softmax

---

⚠️ Limitations

- Depends on image quality
- Less accurate in low lighting
- Limited to selected countries

---

🔮 Future Enhancements

- Add more countries
- Detect currency denominations
- Fake currency detection
- Mobile app development

---

👨‍💻 Author

Lakshmi S B
---

⭐ Note

Dataset, model files, and virtual environment are excluded using .gitignore.
