# ai-painter

# AI Painter (Hand Gesture Drawing)

OpenCV + hand tracking virtual painter controlled by index finger (draw) and index+middle (toolbar select).

## Features
- Dynamic color/eraser selection from header images
- Gesture modes (selection vs drawing)
- Simple smoothing and erasing

## Install
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python "Artificial Painter.py"
```

Place header images in a folder named `header/` beside the script (e.g., 1.jpg, 2.jpg ...).

## hand_track Module
Your `hand_track.py` should expose:
```python
handDetector(detectionCon=0.85)
detector.findHands(img)
detector.findPositions(img, draw=False)
detector.fingersUp()
```

If it uses MediaPipe ensure `mediapipe` is in requirements.

## Packaging (optional)
```bash
pip install pyinstaller
pyinstaller -F "Artificial Painter.py" --name painter
```
