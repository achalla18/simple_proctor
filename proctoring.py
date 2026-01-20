"""
AI Proctoring System - Simple Version
=====================================
A simple exam proctoring system using OpenCV and MediaPipe.

Features:
- Face detection (presence/absence)
- Multiple face detection
- OS-level notifications
- Screenshot capture on violations

Usage:
    python proctoring.py
"""

import cv2
import numpy as np
import time
import os
import threading
from datetime import datetime
from collections import deque
from pathlib import Path

# Check MediaPipe installation
try:
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise ImportError("MediaPipe not properly installed")
    mp_face_detection = mp.solutions.face_detection
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe loaded successfully")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠ MediaPipe not available: {e}")
    print("  Using fallback face detection (Haar Cascades)")

# Check for notification support
NOTIFICATION_AVAILABLE = False
notification_method = None

# Try Windows toast notifications first
try:
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
    NOTIFICATION_AVAILABLE = True
    notification_method = "win10toast"
    print("✓ Windows notifications available (win10toast)")
except ImportError:
    pass

# Try plyer as fallback (cross-platform)
if not NOTIFICATION_AVAILABLE:
    try:
        from plyer import notification as plyer_notification
        NOTIFICATION_AVAILABLE = True
        notification_method = "plyer"
        print("✓ Notifications available (plyer)")
    except ImportError:
        pass

if not NOTIFICATION_AVAILABLE:
    print("⚠ OS notifications not available")
    print("  Install with: pip install win10toast  (Windows)")
    print("  Or: pip install plyer  (Cross-platform)")


def send_notification(title, message, duration=5):
    """Send OS-level notification."""
    if not NOTIFICATION_AVAILABLE:
        return False
    
    try:
        if notification_method == "win10toast":
            # Run in thread to avoid blocking
            threading.Thread(
                target=toaster.show_toast,
                args=(title, message),
                kwargs={'duration': duration, 'threaded': True},
                daemon=True
            ).start()
            return True
        elif notification_method == "plyer":
            plyer_notification.notify(
                title=title,
                message=message,
                timeout=duration
            )
            return True
    except Exception as e:
        print(f"Notification error: {e}")
    return False


class ScreenshotManager:
    """Manages screenshot capture for violations."""
    
    def __init__(self, save_dir="screenshots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.screenshot_count = 0
        self.last_screenshot_time = 0
        self.min_interval = 3.0  # Minimum seconds between screenshots
    
    def capture(self, frame, alert_type, message=""):
        """
        Capture and save a screenshot.
        Returns: filepath if saved, None if skipped (cooldown)
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_screenshot_time < self.min_interval:
            return None
        
        self.last_screenshot_time = current_time
        self.screenshot_count += 1
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{alert_type}_{timestamp}_{self.screenshot_count:04d}.jpg"
        filepath = self.save_dir / filename
        
        # Add overlay text to screenshot
        screenshot = frame.copy()
        h, w = screenshot.shape[:2]
        
        # Add red border
        cv2.rectangle(screenshot, (0, 0), (w-1, h-1), (0, 0, 255), 5)
        
        # Add alert info at bottom
        cv2.rectangle(screenshot, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.putText(screenshot, f"VIOLATION: {alert_type.upper()}", (10, h-35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(screenshot, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save
        cv2.imwrite(str(filepath), screenshot, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"📸 Screenshot saved: {filepath}")
        
        return filepath
    
    def get_count(self):
        """Get total screenshot count."""
        return self.screenshot_count
    
    def get_all_screenshots(self):
        """Get list of all screenshot files."""
        return list(self.save_dir.glob("*.jpg"))


class FaceDetector:
    """Face detection - uses MediaPipe if available, otherwise Haar Cascades."""
    
    def __init__(self):
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            self.detector = mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise RuntimeError("Failed to load Haar Cascade classifier")
    
    def detect(self, frame):
        """
        Detect faces in frame.
        Returns: (face_count, face_boxes)
        """
        if self.use_mediapipe:
            return self._detect_mediapipe(frame)
        else:
            return self._detect_haar(frame)
    
    def _detect_mediapipe(self, frame):
        """MediaPipe face detection."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]
                faces.append({
                    'box': (x, y, width, height),
                    'confidence': confidence
                })
        
        return len(faces), faces
    
    def _detect_haar(self, frame):
        """OpenCV Haar Cascade face detection (fallback)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        
        faces = []
        for (x, y, w, h) in detections:
            faces.append({
                'box': (x, y, w, h),
                'confidence': 0.9
            })
        
        return len(faces), faces


class Alert:
    """Simple alert class."""
    
    def __init__(self, alert_type, message, severity="medium", screenshot_path=None):
        self.type = alert_type
        self.message = message
        self.severity = severity
        self.timestamp = datetime.now()
        self.screenshot_path = screenshot_path
    
    def __str__(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.type}: {self.message}"


class ProctorSystem:
    """Main proctoring system with face detection, notifications, and screenshots."""
    
    def __init__(self, screenshot_dir="screenshots", enable_notifications=True):
        print("Initializing AI Proctoring System...")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.screenshot_manager = ScreenshotManager(screenshot_dir)
        self.enable_notifications = enable_notifications and NOTIFICATION_AVAILABLE
        
        # State tracking
        self.face_absent_since = None
        self.alerts = deque(maxlen=100)
        self.alert_cooldowns = {}
        
        # Thresholds (in seconds)
        self.FACE_ABSENT_THRESHOLD = 3.0
        self.ALERT_COOLDOWN = 5.0
        self.NOTIFICATION_COOLDOWN = 10.0  # Longer cooldown for notifications
        self.notification_cooldowns = {}
        
        # Stats
        self.frame_count = 0
        self.violation_count = 0
        self.session_start = None
        
        print("System ready!")
        if self.enable_notifications:
            print("✓ OS notifications enabled")
        else:
            print("⚠ OS notifications disabled")
    
    def start_session(self):
        """Start a new proctoring session."""
        self.session_start = time.time()
        self.violation_count = 0
        self.alerts.clear()
        self.face_absent_since = None
        self.alert_cooldowns = {}
        self.notification_cooldowns = {}
        
        # Send start notification
        if self.enable_notifications:
            send_notification(
                "Proctoring Started",
                "Your exam session is now being monitored.",
                duration=3
            )
        
        print(f"Session started at {datetime.now().strftime('%H:%M:%S')}")
    
    def process_frame(self, frame):
        """
        Process a single frame and return annotated frame with status.
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Make a copy for annotation
        display = frame.copy()
        h, w = display.shape[:2]
        
        # --- Face Detection ---
        face_count, faces = self.face_detector.detect(frame)
        
        face_status = "OK"
        face_color = (0, 255, 0)  # Green
        
        if face_count == 0:
            # No face detected
            if self.face_absent_since is None:
                self.face_absent_since = current_time
            
            absent_duration = current_time - self.face_absent_since
            
            if absent_duration >= self.FACE_ABSENT_THRESHOLD:
                face_status = f"ABSENT ({absent_duration:.1f}s)"
                face_color = (0, 0, 255)  # Red
                self._add_alert(
                    "face_absence",
                    f"Face not detected for {absent_duration:.1f}s",
                    "high",
                    frame
                )
            else:
                face_status = f"Checking... ({absent_duration:.1f}s)"
                face_color = (0, 255, 255)  # Yellow
        else:
            self.face_absent_since = None
            
            if face_count > 1:
                face_status = f"MULTIPLE FACES ({face_count})"
                face_color = (0, 0, 255)  # Red
                self._add_alert(
                    "multiple_faces",
                    f"Multiple faces detected: {face_count}",
                    "critical",
                    frame
                )
            
            # Draw face boxes
            for face in faces:
                x, y, fw, fh = face['box']
                cv2.rectangle(display, (x, y), (x + fw, y + fh), face_color, 2)
                conf_text = f"{face['confidence']:.0%}"
                cv2.putText(display, conf_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
        
        # --- Draw Status Panel ---
        self._draw_status_panel(display, face_status, face_color)
        
        # --- Draw Alerts ---
        self._draw_alerts(display)
        
        return display
    
    def _add_alert(self, alert_type, message, severity, frame=None):
        """Add an alert with cooldown check, notification, and screenshot."""
        current_time = time.time()
        
        # Check alert cooldown
        last_alert = self.alert_cooldowns.get(alert_type, 0)
        if current_time - last_alert < self.ALERT_COOLDOWN:
            return
        
        # Capture screenshot
        screenshot_path = None
        if frame is not None:
            screenshot_path = self.screenshot_manager.capture(frame, alert_type, message)
        
        # Create and store alert
        alert = Alert(alert_type, message, severity, screenshot_path)
        self.alerts.append(alert)
        self.alert_cooldowns[alert_type] = current_time
        self.violation_count += 1
        
        print(f"⚠️ ALERT: {alert}")
        
        # Send OS notification (with separate cooldown)
        if self.enable_notifications:
            last_notif = self.notification_cooldowns.get(alert_type, 0)
            if current_time - last_notif >= self.NOTIFICATION_COOLDOWN:
                notification_title = "⚠️ Proctoring Alert"
                notification_msg = f"{alert_type.replace('_', ' ').title()}: {message}"
                send_notification(notification_title, notification_msg, duration=5)
                self.notification_cooldowns[alert_type] = current_time
    
    def _draw_status_panel(self, frame, face_status, face_color):
        """Draw status information on frame."""
        h, w = frame.shape[:2]
        
        # Background panel
        cv2.rectangle(frame, (10, 10), (320, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 100), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "AI PROCTOR", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Face status
        cv2.putText(frame, f"Face: {face_status}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
        
        # Screenshot count
        screenshot_count = self.screenshot_manager.get_count()
        cv2.putText(frame, f"Screenshots: {screenshot_count}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Stats panel (bottom)
        if self.session_start:
            elapsed = int(time.time() - self.session_start)
            mins, secs = divmod(elapsed, 60)
            
            cv2.rectangle(frame, (10, h - 50), (220, h - 10), (0, 0, 0), -1)
            cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}", (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Violations: {self.violation_count}", (120, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_alerts(self, frame):
        """Draw recent alerts on frame."""
        h, w = frame.shape[:2]
        
        # Show last 3 alerts
        recent_alerts = list(self.alerts)[-3:]
        
        if recent_alerts:
            panel_height = 30 + len(recent_alerts) * 25
            cv2.rectangle(frame, (w - 380, 10), (w - 10, 10 + panel_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (w - 380, 10), (w - 10, 10 + panel_height), (100, 100, 100), 1)
            cv2.putText(frame, "Recent Alerts:", (w - 370, 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            for i, alert in enumerate(recent_alerts):
                color = (0, 0, 255) if alert.severity == "critical" else \
                        (0, 165, 255) if alert.severity == "high" else (0, 200, 255)
                text = f"{alert.timestamp.strftime('%H:%M:%S')} - {alert.type}"
                cv2.putText(frame, text, (w - 370, 52 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Show screenshot indicator
                if alert.screenshot_path:
                    cv2.putText(frame, "[SS]", (w - 50, 52 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
    
    def get_stats(self):
        """Get current session statistics."""
        elapsed = 0
        if self.session_start:
            elapsed = time.time() - self.session_start
        
        return {
            'elapsed_seconds': int(elapsed),
            'violation_count': self.violation_count,
            'frame_count': self.frame_count,
            'alert_count': len(self.alerts),
            'screenshot_count': self.screenshot_manager.get_count()
        }
    
    def get_screenshots(self):
        """Get list of all screenshot files."""
        return self.screenshot_manager.get_all_screenshots()


def run_cli():
    """Run the proctoring system with webcam."""
    print("=" * 50)
    print("AI Proctoring System - CLI Mode")
    print("=" * 50)
    print("\nControls:")
    print("  Q - Quit")
    print("  R - Reset session")
    print("  S - Take manual screenshot")
    print("  N - Test notification")
    print()
    
    # Initialize
    proctor = ProctorSystem(screenshot_dir="screenshots", enable_notifications=True)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        print("Make sure your webcam is connected and not used by another app.")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Webcam opened successfully!")
    proctor.start_session()
    
    last_frame = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            last_frame = frame.copy()
            
            # Process frame
            display = proctor.process_frame(frame)
            
            # Show frame
            cv2.imshow('AI Proctoring System', display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                proctor.start_session()
                print("Session reset!")
            elif key == ord('s'):
                # Manual screenshot
                if last_frame is not None:
                    path = proctor.screenshot_manager.capture(last_frame, "manual", "Manual capture")
                    if path:
                        print(f"Manual screenshot saved: {path}")
            elif key == ord('n'):
                # Test notification
                if send_notification("Test Notification", "This is a test notification from AI Proctor"):
                    print("Test notification sent!")
                else:
                    print("Notifications not available")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = proctor.get_stats()
        print("\n" + "=" * 50)
        print("Session Summary")
        print("=" * 50)
        print(f"Duration: {stats['elapsed_seconds'] // 60}m {stats['elapsed_seconds'] % 60}s")
        print(f"Total Violations: {stats['violation_count']}")
        print(f"Screenshots Captured: {stats['screenshot_count']}")
        print(f"Frames Processed: {stats['frame_count']}")
        
        # List screenshots
        screenshots = proctor.get_screenshots()
        if screenshots:
            print(f"\nScreenshots saved in 'screenshots' folder:")
            for ss in screenshots[-5:]:  # Show last 5
                print(f"  - {ss.name}")
            if len(screenshots) > 5:
                print(f"  ... and {len(screenshots) - 5} more")


if __name__ == "__main__":
    run_cli()