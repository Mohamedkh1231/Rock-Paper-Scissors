# Rock Paper Scissors AI Game 🎮🤖

An interactive, real-time Rock Paper Scissors game powered by a Convolutional Neural Network (CNN) and OpenCV. The application recognizes hand gestures using your webcam or DroidCam and lets you play against an AI opponent in real time.

---

## 📌 Features

* **Real-time Gesture Recognition:** Detects Rock, Paper, and Scissors gestures instantly using deep learning.
* **Focused ROI Processing:** Uses a central Region of Interest (ROI) box to optimize computer vision predictions.
* **Game State Management:** Handles transitions smoothly between Waiting, Countdown, and Result states.
* **Live Scoreboard:** Displays real-time scores (Player vs. AI, Draws), round count, and prediction confidence on the screen.

---

## 🛠️ Requirements

Ensure you have **Python 3.8+** installed before running the project.

### Dependencies

* `opencv-python`
* `tensorflow`
* `numpy`

---

## 🚀 Installation & Setup

### 1. Install Dependencies

Run the following command in your terminal:

```bash
pip install -r requirements.txt
