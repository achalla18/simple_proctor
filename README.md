# Machine Learning based Proctoring System - Simple Version

A simple, working Machine Learning based proctoring system for exam monitoring.

## Features

- Face detection (checks if your face is visible)
- Multiple face detection (alerts if more than one person)
- Real-time alerts and statistics
- Web interface with live video

## Requirements

- Python 3.8 or higher
- Webcam
- Windows/Mac/Linux

## Installation

1. **Open Command Prompt/Terminal** in the project folder

2. **Install dependencies:**
```bash
pip install opencv-python flask numpy
```


## Usage

### Option 1: Command Line Mode (Simplest)

```bash
python proctoring.py
```

This opens a window with your webcam feed showing:
- Face detection boxes
- Gaze status
- Real-time alerts


### Option 2: Web Interface

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

## How It Works

### Face Detection
- Uses OpenCV Haar Cascades to detect faces in real-time
- Alerts after 3 seconds if no face is detected
- Alerts immediately if multiple faces are detected



## File Structure

```
ai_proctoring_system/
├── proctor.py      # Main detection code (run this for CLI)
├── app.py         # Web interface server
├── templates/
│   └── index.html     # Web page
├── requirements.txt   # Dependencies
└── README.md          # This file
```