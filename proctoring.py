
import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque

# Check MediaPipe installation
try:
    import mediapipe as mp
    # Verify mediapipe is properly installed
    if not hasattr(mp, 'solutions'):
        raise ImportError("MediaPipe not properly installed")
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe loaded successfully")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠ MediaPipe not available: {e}")
    print("  Run: pip uninstall mediapipe && pip install mediapipe")
    print("  Using fallback face detection (Haar Cascades)")


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
            # Fallback to OpenCV Haar Cascades
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
                'confidence': 0.9  # Haar doesn't provide confidence
            })
        
        return len(faces), faces


class GazeDetector:
    """
    Enhanced gaze detection using MediaPipe Face Mesh.
    
    Combines:
    - Iris position tracking (where eyes are looking)
    - Head pose estimation (which way head is turned)
    - Smoothing to reduce false positives
    """
    
    def __init__(self):
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Eye landmark indices
            self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
            self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
            self.LEFT_IRIS = [474, 475, 476, 477]
            self.RIGHT_IRIS = [469, 470, 471, 472]
            
            # Head pose landmarks (nose tip, chin, left eye corner, right eye corner, etc.)
            self.NOSE_TIP = 1
            self.CHIN = 152
            self.LEFT_EYE_CORNER = 263
            self.RIGHT_EYE_CORNER = 33
            self.LEFT_MOUTH = 287
            self.RIGHT_MOUTH = 57
        
        # Smoothing - keep history of recent detections
        self.history_size = 5
        self.gaze_history = []
        self.head_pose_history = []
        
        # Thresholds (can be adjusted)
        self.IRIS_HORIZONTAL_THRESHOLD = 0.32  # How far iris can move left/right
        self.IRIS_VERTICAL_THRESHOLD = 0.38    # How far iris can move up/down
        self.HEAD_YAW_THRESHOLD = 20           # Degrees head can turn left/right
        self.HEAD_PITCH_THRESHOLD = 15         # Degrees head can tilt up/down
        
        # Last known good values
        self.last_direction = "center"
        self.last_confidence = 0.0
    
    def detect(self, frame):
        """
        Detect gaze direction with enhanced accuracy.
        
        Returns: (gaze_detected, is_looking_center, direction, confidence, details)
        - gaze_detected: bool - whether a face was found
        - is_looking_center: bool - whether looking at screen
        - direction: str - "center", "left", "right", "up", "down", "away"
        - confidence: float - how confident we are (0-1)
        - details: dict - additional info for visualization
        """
        if not self.use_mediapipe:
            return False, True, "unavailable", 0.0, {}
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return False, False, "no_face", 0.0, {}
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        details = {}
        
        try:
            # 1. Get iris-based gaze direction
            iris_direction, iris_offset = self._get_iris_direction(landmarks, w, h)
            details['iris_direction'] = iris_direction
            details['iris_offset'] = iris_offset
            
            # 2. Get head pose
            head_yaw, head_pitch, head_roll = self._get_head_pose(landmarks, w, h)
            details['head_yaw'] = head_yaw
            details['head_pitch'] = head_pitch
            details['head_roll'] = head_roll
            
            # 3. Combine iris and head pose for final direction
            direction, confidence = self._combine_signals(
                iris_direction, iris_offset,
                head_yaw, head_pitch
            )
            
            # 4. Apply smoothing
            direction, confidence = self._smooth_detection(direction, confidence)
            
            details['raw_direction'] = direction
            details['confidence'] = confidence
            
            # 5. Determine if looking at center
            is_center = direction == "center" or confidence < 0.4
            
            # Store for next frame
            self.last_direction = direction
            self.last_confidence = confidence
            
            return True, is_center, direction, confidence, details
            
        except Exception as e:
            return True, True, "center", 0.5, {'error': str(e)}
    
    def _get_iris_direction(self, landmarks, w, h):
        """
        Calculate gaze direction based on iris position within eye.
        Returns: (direction, offset_magnitude)
        """
        left_iris = self._get_iris_position(landmarks, self.LEFT_IRIS, self.LEFT_EYE, w, h)
        right_iris = self._get_iris_position(landmarks, self.RIGHT_IRIS, self.RIGHT_EYE, w, h)
        
        if left_iris is None or right_iris is None:
            return "center", (0.5, 0.5)
        
        # Average both eyes
        avg_h = (left_iris[0] + right_iris[0]) / 2
        avg_v = (left_iris[1] + right_iris[1]) / 2
        
        # Calculate offset from center (0.5, 0.5)
        h_offset = avg_h - 0.5  # Negative = looking left, Positive = looking right
        v_offset = avg_v - 0.5  # Negative = looking up, Positive = looking down
        
        direction = "center"
        
        # Check horizontal first (usually more significant)
        if h_offset < -self.IRIS_HORIZONTAL_THRESHOLD:
            direction = "left"
        elif h_offset > self.IRIS_HORIZONTAL_THRESHOLD:
            direction = "right"
        # Then vertical
        elif v_offset < -self.IRIS_VERTICAL_THRESHOLD:
            direction = "up"
        elif v_offset > self.IRIS_VERTICAL_THRESHOLD:
            direction = "down"
        
        return direction, (h_offset, v_offset)
    
    def _get_iris_position(self, landmarks, iris_indices, eye_indices, w, h):
        """Calculate iris position ratio within eye bounds (0-1 scale)."""
        try:
            eye_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
            eye_x = [p[0] for p in eye_points]
            eye_y = [p[1] for p in eye_points]
            
            iris_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices]
            iris_cx = sum(p[0] for p in iris_points) / len(iris_points)
            iris_cy = sum(p[1] for p in iris_points) / len(iris_points)
            
            x_range = max(eye_x) - min(eye_x)
            y_range = max(eye_y) - min(eye_y)
            
            if x_range < 5 or y_range < 5:  # Eye too small/closed
                return None
            
            x_ratio = (iris_cx - min(eye_x)) / x_range
            y_ratio = (iris_cy - min(eye_y)) / y_range
            
            # Clamp to valid range
            x_ratio = max(0, min(1, x_ratio))
            y_ratio = max(0, min(1, y_ratio))
            
            return (x_ratio, y_ratio)
        except:
            return None
    
    def _get_head_pose(self, landmarks, w, h):
        """
        Estimate head pose (yaw, pitch, roll) from facial landmarks.
        Returns angles in degrees.
        """
        try:
            # Get key facial points
            nose = np.array([landmarks[self.NOSE_TIP].x * w, 
                           landmarks[self.NOSE_TIP].y * h,
                           landmarks[self.NOSE_TIP].z * w])
            
            chin = np.array([landmarks[self.CHIN].x * w,
                           landmarks[self.CHIN].y * h,
                           landmarks[self.CHIN].z * w])
            
            left_eye = np.array([landmarks[self.LEFT_EYE_CORNER].x * w,
                                landmarks[self.LEFT_EYE_CORNER].y * h,
                                landmarks[self.LEFT_EYE_CORNER].z * w])
            
            right_eye = np.array([landmarks[self.RIGHT_EYE_CORNER].x * w,
                                 landmarks[self.RIGHT_EYE_CORNER].y * h,
                                 landmarks[self.RIGHT_EYE_CORNER].z * w])
            
            left_mouth = np.array([landmarks[self.LEFT_MOUTH].x * w,
                                  landmarks[self.LEFT_MOUTH].y * h,
                                  landmarks[self.LEFT_MOUTH].z * w])
            
            right_mouth = np.array([landmarks[self.RIGHT_MOUTH].x * w,
                                   landmarks[self.RIGHT_MOUTH].y * h,
                                   landmarks[self.RIGHT_MOUTH].z * w])
            
            # Calculate face center
            face_center = (left_eye + right_eye + left_mouth + right_mouth) / 4
            
            # Yaw (left-right rotation) - based on nose offset from face center
            # and asymmetry between eye distances
            eye_center = (left_eye + right_eye) / 2
            nose_offset_x = nose[0] - eye_center[0]
            left_dist = np.linalg.norm(nose[:2] - left_eye[:2])
            right_dist = np.linalg.norm(nose[:2] - right_eye[:2])
            
            # Normalize and convert to degrees
            eye_width = np.linalg.norm(left_eye[:2] - right_eye[:2])
            if eye_width > 0:
                yaw = (nose_offset_x / eye_width) * 60  # Scale to ~degrees
                yaw += ((right_dist - left_dist) / eye_width) * 30  # Adjustment
            else:
                yaw = 0
            
            # Pitch (up-down rotation) - based on nose-to-eye vs nose-to-chin ratio
            nose_to_eye = np.linalg.norm(nose[:2] - eye_center[:2])
            nose_to_chin = np.linalg.norm(nose[:2] - chin[:2])
            
            if nose_to_chin > 0:
                pitch_ratio = nose_to_eye / nose_to_chin
                pitch = (pitch_ratio - 0.5) * 60  # Scale to ~degrees
            else:
                pitch = 0
            
            # Roll (tilt) - based on eye line angle
            eye_delta = right_eye - left_eye
            roll = np.degrees(np.arctan2(eye_delta[1], eye_delta[0]))
            
            return float(yaw), float(pitch), float(roll)
            
        except Exception as e:
            return 0.0, 0.0, 0.0
    
    def _combine_signals(self, iris_direction, iris_offset, head_yaw, head_pitch):
        """
        Combine iris direction and head pose for final gaze direction.
        Returns: (direction, confidence)
        """
        h_offset, v_offset = iris_offset
        
        # Start with iris-based direction
        direction = iris_direction
        confidence = 0.0
        
        # Calculate confidence based on how far from center
        iris_confidence = max(abs(h_offset), abs(v_offset)) / 0.5  # 0-1 scale
        iris_confidence = min(1.0, iris_confidence)
        
        # Head pose contribution
        head_direction = "center"
        head_confidence = 0.0
        
        if abs(head_yaw) > self.HEAD_YAW_THRESHOLD:
            head_direction = "left" if head_yaw < 0 else "right"
            head_confidence = min(1.0, abs(head_yaw) / 45)
        elif abs(head_pitch) > self.HEAD_PITCH_THRESHOLD:
            head_direction = "up" if head_pitch < 0 else "down"
            head_confidence = min(1.0, abs(head_pitch) / 30)
        
        # Combine: if both agree, high confidence; if they disagree, lower confidence
        if iris_direction == head_direction:
            direction = iris_direction
            confidence = (iris_confidence + head_confidence) / 2 + 0.2  # Boost for agreement
        elif iris_direction != "center" and head_direction == "center":
            direction = iris_direction
            confidence = iris_confidence * 0.7
        elif head_direction != "center" and iris_direction == "center":
            direction = head_direction
            confidence = head_confidence * 0.7
        elif iris_direction != "center" and head_direction != "center":
            # Both indicate looking away but in different directions
            # Trust the one with higher confidence
            if iris_confidence > head_confidence:
                direction = iris_direction
                confidence = iris_confidence * 0.8
            else:
                direction = head_direction
                confidence = head_confidence * 0.8
        else:
            direction = "center"
            confidence = 1.0 - max(iris_confidence, head_confidence)
        
        # If head is significantly turned, it's "away" regardless of iris
        if abs(head_yaw) > 35 or abs(head_pitch) > 25:
            direction = "away"
            confidence = 0.9
        
        return direction, min(1.0, confidence)
    
    def _smooth_detection(self, direction, confidence):
        """
        Apply temporal smoothing to reduce jitter and false positives.
        """
        # Add to history
        self.gaze_history.append((direction, confidence))
        
        # Keep only recent history
        if len(self.gaze_history) > self.history_size:
            self.gaze_history.pop(0)
        
        # If we don't have enough history, return current
        if len(self.gaze_history) < 3:
            return direction, confidence
        
        # Count directions in history
        direction_counts = {}
        total_confidence = 0
        
        for d, c in self.gaze_history:
            direction_counts[d] = direction_counts.get(d, 0) + 1
            total_confidence += c
        
        # Find most common direction
        most_common = max(direction_counts, key=direction_counts.get)
        count = direction_counts[most_common]
        
        # Only change from center if we have consistent non-center readings
        if self.last_direction == "center" and most_common != "center":
            # Need majority to change away from center
            if count >= len(self.gaze_history) * 0.6:
                return most_common, total_confidence / len(self.gaze_history)
            else:
                return "center", 0.3
        
        # If currently looking away, can return to center faster
        elif self.last_direction != "center" and most_common == "center":
            if count >= len(self.gaze_history) * 0.4:
                return "center", 0.5
            else:
                return self.last_direction, total_confidence / len(self.gaze_history)
        
        return most_common, total_confidence / len(self.gaze_history)


class Alert:
    """Simple alert class."""
    
    def __init__(self, alert_type, message, severity="medium"):
        self.type = alert_type
        self.message = message
        self.severity = severity
        self.timestamp = datetime.now()
    
    def __str__(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.type}: {self.message}"


class ProctorSystem:
    """Main proctoring system that combines all detectors."""
    
    def __init__(self):
        print("Initializing AI Proctoring System...")
        
        # Initialize detectors
        self.face_detector = FaceDetector()
        self.gaze_detector = GazeDetector()
        
        # State tracking
        self.face_absent_since = None
        self.gaze_away_since = None
        self.alerts = deque(maxlen=100)
        self.alert_cooldowns = {}
        
        # Thresholds (in seconds)
        self.FACE_ABSENT_THRESHOLD = 3.0
        self.GAZE_AWAY_THRESHOLD = 2.0
        self.ALERT_COOLDOWN = 5.0
        
        # Stats
        self.frame_count = 0
        self.violation_count = 0
        self.session_start = None
        
        # Gaze visualization
        self.gaze_indicator_pos = None
        
        print("System ready!")
    
    def start_session(self):
        """Start a new proctoring session."""
        self.session_start = time.time()
        self.violation_count = 0
        self.alerts.clear()
        self.face_absent_since = None
        self.gaze_away_since = None
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
                self._add_alert("face_absence", f"Face not detected for {absent_duration:.1f}s", "high")
            else:
                face_status = f"Checking... ({absent_duration:.1f}s)"
                face_color = (0, 255, 255)  # Yellow
        else:
            self.face_absent_since = None
            
            if face_count > 1:
                face_status = f"MULTIPLE ({face_count})"
                face_color = (0, 0, 255)  # Red
                self._add_alert("multiple_faces", f"Multiple faces detected: {face_count}", "critical")
            
            # Draw face boxes
            for face in faces:
                x, y, fw, fh = face['box']
                cv2.rectangle(display, (x, y), (x + fw, y + fh), face_color, 2)
                conf_text = f"{face['confidence']:.0%}"
                cv2.putText(display, conf_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
        
        # --- Enhanced Gaze Detection ---
        gaze_result = self.gaze_detector.detect(frame)
        gaze_detected, is_center, direction, confidence, details = gaze_result
        
        gaze_status = "OK"
        gaze_color = (0, 255, 0)  # Green
        
        if gaze_detected:
            if not is_center:
                if self.gaze_away_since is None:
                    self.gaze_away_since = current_time
                
                away_duration = current_time - self.gaze_away_since
                
                if away_duration >= self.GAZE_AWAY_THRESHOLD:
                    if direction == "away":
                        gaze_status = f"LOOKING AWAY ({away_duration:.1f}s)"
                    else:
                        gaze_status = f"LOOKING {direction.upper()} ({away_duration:.1f}s)"
                    gaze_color = (0, 0, 255)  # Red
                    self._add_alert("gaze_deviation", 
                                  f"Looking {direction} for {away_duration:.1f}s (conf: {confidence:.0%})", 
                                  "medium")
                else:
                    gaze_status = f"Looking {direction}... ({confidence:.0%})"
                    gaze_color = (0, 255, 255)  # Yellow
            else:
                self.gaze_away_since = None
                gaze_status = f"CENTER ({confidence:.0%})"
        else:
            gaze_status = "No face"
            gaze_color = (128, 128, 128)  # Gray
        
        # --- Draw Enhanced Gaze Visualization ---
        if gaze_detected and faces:
            self._draw_gaze_indicator(display, direction, confidence, details, faces[0])
        
        # --- Draw Status Panel ---
        self._draw_status_panel(display, face_status, face_color, gaze_status, gaze_color, details)
        
        # --- Draw Alerts ---
        self._draw_alerts(display)
        
        return display
    
    def _draw_gaze_indicator(self, frame, direction, confidence, details, face):
        """Draw visual indicator showing where the person is looking."""
        h, w = frame.shape[:2]
        x, y, fw, fh = face['box']
        
        # Face center
        face_cx = x + fw // 2
        face_cy = y + fh // 2
        
        # Draw gaze direction arrow from face
        arrow_length = int(min(fw, fh) * 0.5 * confidence)
        
        # Direction vectors
        dir_vectors = {
            'left': (-1, 0),
            'right': (1, 0),
            'up': (0, -1),
            'down': (0, 1),
            'center': (0, 0),
            'away': (-0.7, -0.7) if details.get('head_yaw', 0) < 0 else (0.7, -0.7)
        }
        
        dx, dy = dir_vectors.get(direction, (0, 0))
        
        if direction == "center":
            # Draw a circle at face center for "looking at camera"
            cv2.circle(frame, (face_cx, face_cy), 15, (0, 255, 0), 3)
            cv2.circle(frame, (face_cx, face_cy), 5, (0, 255, 0), -1)
        else:
            # Draw arrow showing gaze direction
            end_x = int(face_cx + dx * arrow_length)
            end_y = int(face_cy + dy * arrow_length)
            
            color = (0, 0, 255) if confidence > 0.5 else (0, 255, 255)
            cv2.arrowedLine(frame, (face_cx, face_cy), (end_x, end_y), 
                           color, 3, tipLength=0.3)
        
        # Draw head pose indicator (small)
        if 'head_yaw' in details and 'head_pitch' in details:
            yaw = details['head_yaw']
            pitch = details['head_pitch']
            
            # Draw mini head orientation indicator in corner
            indicator_x = w - 80
            indicator_y = 150
            indicator_size = 30
            
            # Background circle
            cv2.circle(frame, (indicator_x, indicator_y), indicator_size + 5, (50, 50, 50), -1)
            cv2.circle(frame, (indicator_x, indicator_y), indicator_size + 5, (100, 100, 100), 2)
            
            # Head direction dot
            dot_x = int(indicator_x + (yaw / 45) * indicator_size)
            dot_y = int(indicator_y + (pitch / 30) * indicator_size)
            
            # Center zone
            cv2.circle(frame, (indicator_x, indicator_y), 10, (0, 100, 0), 1)
            
            # Current position
            dot_color = (0, 255, 0) if abs(yaw) < 20 and abs(pitch) < 15 else (0, 0, 255)
            cv2.circle(frame, (dot_x, dot_y), 6, dot_color, -1)
            
            cv2.putText(frame, "Head", (indicator_x - 20, indicator_y + indicator_size + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _add_alert(self, alert_type, message, severity):
        """Add an alert with cooldown check."""
        current_time = time.time()
        
        # Check cooldown
        last_alert = self.alert_cooldowns.get(alert_type, 0)
        if current_time - last_alert < self.ALERT_COOLDOWN:
            return
        
        # Create and store alert
        alert = Alert(alert_type, message, severity)
        self.alerts.append(alert)
        self.alert_cooldowns[alert_type] = current_time
        self.violation_count += 1
        
        print(f"⚠️ ALERT: {alert}")
    
    def _draw_status_panel(self, frame, face_status, face_color, gaze_status, gaze_color, details=None):
        """Draw status information on frame."""
        h, w = frame.shape[:2]
        
        # Background panel
        cv2.rectangle(frame, (10, 10), (320, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 110), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "AI PROCTOR", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Face status
        cv2.putText(frame, f"Face: {face_status}", (20, 58), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
        
        # Gaze status
        cv2.putText(frame, f"Gaze: {gaze_status}", (20, 78), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)
        
        # Head pose info (if available)
        if details and 'head_yaw' in details:
            yaw = details.get('head_yaw', 0)
            pitch = details.get('head_pitch', 0)
            pose_text = f"Head: Y:{yaw:+.0f} P:{pitch:+.0f}"
            pose_color = (0, 255, 0) if abs(yaw) < 20 and abs(pitch) < 15 else (0, 165, 255)
            cv2.putText(frame, pose_text, (20, 98), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, pose_color, 1)
        
        # Stats panel (bottom)
        if self.session_start:
            elapsed = int(time.time() - self.session_start)
            mins, secs = divmod(elapsed, 60)
            
            cv2.rectangle(frame, (10, h - 50), (200, h - 10), (0, 0, 0), -1)
            cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}", (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Violations: {self.violation_count}", (20, h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_alerts(self, frame):
        """Draw recent alerts on frame."""
        h, w = frame.shape[:2]
        
        # Show last 3 alerts
        recent_alerts = list(self.alerts)[-3:]
        
        if recent_alerts:
            panel_height = 30 + len(recent_alerts) * 22
            cv2.rectangle(frame, (w - 350, 10), (w - 10, 10 + panel_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (w - 350, 10), (w - 10, 10 + panel_height), (100, 100, 100), 1)
            cv2.putText(frame, "Recent Alerts:", (w - 340, 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            for i, alert in enumerate(recent_alerts):
                color = (0, 0, 255) if alert.severity == "critical" else \
                        (0, 165, 255) if alert.severity == "high" else (0, 200, 255)
                text = f"{alert.timestamp.strftime('%H:%M:%S')} - {alert.type}"
                cv2.putText(frame, text, (w - 340, 50 + i * 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def get_stats(self):
        """Get current session statistics."""
        elapsed = 0
        if self.session_start:
            elapsed = time.time() - self.session_start
        
        return {
            'elapsed_seconds': int(elapsed),
            'violation_count': self.violation_count,
            'frame_count': self.frame_count,
            'alert_count': len(self.alerts)
        }


def run_cli():
    """Run the proctoring system with webcam."""
    print("=" * 50)
    print("AI Proctoring System - CLI Mode")
    print("=" * 50)
    print("\nControls:")
    print("  Q - Quit")
    print("  R - Reset session")
    print()
    
    # Initialize
    proctor = ProctorSystem()
    
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
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
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
        print(f"Frames Processed: {stats['frame_count']}")


if __name__ == "__main__":
    run_cli()
