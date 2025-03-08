# Virtual Whiteboard with Hand Tracking

## 📌 Project Overview
This project is a **Virtual Whiteboard** that allows users to draw, erase, and change colors using **hand gestures**. It utilizes **MediaPipe** for hand tracking and **OpenCV** for real-time video processing.

---

## 🎥 Demo
(Add a GIF or Screenshot of the application in action)

---

## 🛠 Features
✔️ **Hand Gesture-Based Drawing** - Use finger gestures to draw on the virtual whiteboard.
✔️ **Color Changing** - Change pen color using specific finger gestures.
✔️ **Eraser Mode** - Use gestures to erase parts of the canvas.
✔️ **Clear Canvas** - Press 'C' to reset the whiteboard.
✔️ **Exit the Program** - Press 'Q' to quit.

---

## 📂 Project Structure
```
📂 Whiteboard
│── 📜 Project.py       # Main Python script
│── 📜 README.md        # Documentation file
```

---

## ⚙️ Installation & Setup

### 1️⃣ Install Dependencies
```bash
pip install opencv-python mediapipe numpy
```

### 2️⃣ Run the Application
```bash
python Project.py
```

---

## 🖐 Gesture Controls
- **Drawing Mode**: Touch **thumb + index/middle/ring finger** to draw.
- **Eraser Mode**: Touch **thumb + pinky** to erase.
- **Change Color**:
  - **Red** → Thumb + Index finger
  - **Green** → Thumb + Middle finger
  - **Blue** → Thumb + Ring finger
- **Clear Screen**: Press `C`
- **Exit**: Press `Q`

---

## 🏗 Future Improvements
- Add more gesture-based controls.
- Implement a GUI for color and thickness selection.
- Improve tracking accuracy.

---

## 📜 License
This project is licensed under the **MIT License**.

---

🎨 Happy Drawing! ✍️

