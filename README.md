# AI Proctoring System - Simple Version

A simple, working AI proctoring system for exam monitoring.

## Features

- ✅ Face detection (checks if your face is visible)
- ✅ Multiple face detection (alerts if more than one person)
- ✅ **Enhanced gaze tracking** with:
  - Iris position tracking (where your eyes are looking)
  - Head pose estimation (which way your head is turned)
  - Smoothing to reduce false alerts
  - Visual indicators showing gaze direction
- ✅ Real-time alerts and statistics
- ✅ Web interface with live video

## Requirements

- Python 3.8 or higher
- Webcam
- Windows/Mac/Linux

## Installation

1. **Open Command Prompt/Terminal** in the project folder

2. **Install dependencies:**
```bash
pip install opencv-python mediapipe flask numpy
```

3. **Run the test script to verify everything works:**
```bash
python setup_test.py
```

## Fixing MediaPipe Issues

If you get "module 'mediapipe' has no attribute 'solutions'" error:

```bash
# Uninstall and reinstall MediaPipe
pip uninstall mediapipe -y
pip install mediapipe
```

If that doesn't work, try:
```bash
# Clean reinstall all packages
pip uninstall opencv-python mediapipe flask numpy -y
pip install opencv-python mediapipe flask numpy
```

**Note:** The system will still work without MediaPipe - it falls back to OpenCV's Haar Cascade for face detection (gaze tracking requires MediaPipe though).

## Usage

### Option 1: Command Line Mode (Simplest)

```bash
python proctoring.py
```

This opens a window with your webcam feed showing:
- Face detection boxes
- Gaze status
- Real-time alerts

**Controls:**
- `Q` - Quit
- `R` - Reset session

### Option 2: Web Interface

```bash
python web_app.py
```

Then open http://localhost:5000 in your browser.

## How It Works

### Face Detection
- Uses MediaPipe to detect faces in real-time
- Falls back to OpenCV Haar Cascades if MediaPipe fails
- Alerts after 3 seconds if no face is detected
- Alerts immediately if multiple faces are detected

### Enhanced Gaze Tracking
The gaze detection combines **two methods** for better accuracy:

1. **Iris Tracking**: Tracks where your eyes are looking by measuring iris position within the eye socket
2. **Head Pose Estimation**: Calculates which way your head is turned (yaw, pitch, roll)

**How it decides you're looking away:**
- If iris moves significantly to one side (left/right/up/down)
- If head turns more than ~20° left/right or ~15° up/down
- If both signals agree → high confidence alert
- If signals disagree → lower confidence, less likely to alert

**Smoothing:**
- Keeps history of last 5 frames
- Requires consistent looking-away readings before alerting
- Prevents false positives from brief glances
- Returns to "center" faster when you look back

### Visual Indicators
- Green circle on face = looking at camera ✓
- Red arrow = direction you're looking
- Head indicator (top-right) shows head orientation
- Yellow = warning, Red = alert

### Alerts
- 2-second threshold before gaze alerts
- 3-second threshold before face absence alerts  
- 5-second cooldown between same alert types
- Different severity levels (medium, high, critical)

## Troubleshooting

### "Cannot open webcam"
- Make sure webcam is connected
- Close other apps using the camera (Zoom, Teams, etc.)
- Try unplugging and reconnecting the webcam

### "ModuleNotFoundError"
Run: `pip install -r requirements.txt`

### Slow performance
- Close other applications
- Reduce screen resolution
- Use CLI mode instead of web interface

## File Structure

```
ai_proctoring_system/
├── proctoring.py      # Main detection code (run this for CLI)
├── web_app.py         # Web interface server
├── templates/
│   └── index.html     # Web page
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Customization

You can adjust these settings in `proctoring.py`:

**Alert Thresholds** (in `ProctorSystem.__init__()`):
```python
self.FACE_ABSENT_THRESHOLD = 3.0   # Seconds before face absence alert
self.GAZE_AWAY_THRESHOLD = 2.0     # Seconds before gaze alert
self.ALERT_COOLDOWN = 5.0          # Seconds between same alerts
```

**Gaze Sensitivity** (in `GazeDetector.__init__()`):
```python
self.IRIS_HORIZONTAL_THRESHOLD = 0.32  # Lower = more sensitive to left/right
self.IRIS_VERTICAL_THRESHOLD = 0.38    # Lower = more sensitive to up/down
self.HEAD_YAW_THRESHOLD = 20           # Degrees before head turn triggers
self.HEAD_PITCH_THRESHOLD = 15         # Degrees before head tilt triggers
```

**Make it less sensitive** (fewer false alerts):
- Increase thresholds (e.g., `IRIS_HORIZONTAL_THRESHOLD = 0.40`)
- Increase `HEAD_YAW_THRESHOLD` to 25-30

**Make it more sensitive** (catch more looking away):
- Decrease thresholds (e.g., `IRIS_HORIZONTAL_THRESHOLD = 0.25`)
- Decrease `HEAD_YAW_THRESHOLD` to 15

## License

Free to use for educational purposes.
