#
# Setup and Test Script for AI Proctoring System

import sys
import subprocess

def check_python():
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠ Warning: Python 3.8+ recommended")
        return False
    print("✓ Python version OK")
    return True

def check_opencv():
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Check if Haar cascades are available
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            print("  ⚠ Haar cascades not found")
        else:
            print("  ✓ Haar cascades available (fallback face detection)")
        return True
    except ImportError:
        print("✗ OpenCV not installed")
        print("  Run: pip install opencv-python")
        return False

def check_mediapipe():
    try:
        import mediapipe as mp
        
        # Check if solutions attribute exists
        if not hasattr(mp, 'solutions'):
            print("✗ MediaPipe installed but broken")
            print("  Run: pip uninstall mediapipe -y && pip install mediapipe")
            return False
        
        # Try to access face detection
        face_det = mp.solutions.face_detection
        face_mesh = mp.solutions.face_mesh
        
        print(f"✓ MediaPipe version: {mp.__version__}")
        return True
        
    except ImportError:
        print("✗ MediaPipe not installed")
        print("  Run: pip install mediapipe")
        return False
    except Exception as e:
        print(f"✗ MediaPipe error: {e}")
        print("  Run: pip uninstall mediapipe -y && pip install mediapipe")
        return False

def check_flask():
    try:
        import flask
        print(f"✓ Flask version: {flask.__version__}")
        return True
    except ImportError:
        print("✗ Flask not installed")
        print("  Run: pip install flask")
        return False

def check_numpy():
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        return True
    except ImportError:
        print("✗ NumPy not installed")
        print("  Run: pip install numpy")
        return False

def check_webcam():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"✓ Webcam working (frame size: {frame.shape[1]}x{frame.shape[0]})")
                return True
        print("✗ Webcam not available or in use")
        print("  Make sure webcam is connected and not used by another app")
        return False
    except Exception as e:
        print(f"✗ Webcam error: {e}")
        return False

def test_face_detection():
    try:
        import cv2
        from proctoring import FaceDetector
        
        print("\nTesting face detection...")
        detector = FaceDetector()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Cannot open webcam for testing")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("✗ Cannot read frame")
            return False
        
        count, faces = detector.detect(frame)
        print(f"✓ Face detection working - detected {count} face(s)")
        return True
        
    except Exception as e:
        print(f"✗ Face detection error: {e}")
        return False

def main():
    print("=" * 50)
    print("AI Proctoring System - Setup Test")
    print("=" * 50)
    print()
    
    all_ok = True
    
    print("Checking dependencies...")
    print("-" * 30)
    
    all_ok &= check_python()
    all_ok &= check_opencv()
    all_ok &= check_numpy()
    all_ok &= check_mediapipe()
    all_ok &= check_flask()
    
    print()
    print("Checking hardware...")
    print("-" * 30)
    
    webcam_ok = check_webcam()
    
    if all_ok and webcam_ok:
        print()
        print("Testing detection...")
        print("-" * 30)
        test_face_detection()
    
    print()
    print("=" * 50)
    
    if all_ok and webcam_ok:
        print("✓ All checks passed! You can run:")
        print()
        print("  python proctoring.py      # CLI mode")
        print("  python web_app.py         # Web interface")
    else:
        print("⚠ Some checks failed. Fix the issues above first.")
        print()
        print("Quick fix - reinstall all dependencies:")
        print("  pip uninstall opencv-python mediapipe flask numpy -y")
        print("  pip install opencv-python mediapipe flask numpy")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
