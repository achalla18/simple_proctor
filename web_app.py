#open http://localhost:5000 in browser


import cv2
import time
import threading
from datetime import datetime
from flask import Flask, Response, render_template, jsonify

# Import the proctoring module
from proctoring import ProctorSystem

app = Flask(__name__)

# Global state
proctor = None
camera = None
is_running = False
frame_lock = threading.Lock()
current_frame = None


def init_system():
    global proctor, camera, is_running
    
    proctor = ProctorSystem()
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("ERROR: Cannot open webcam!")
        return False
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    proctor.start_session()
    is_running = True
    
    # Start processing thread
    thread = threading.Thread(target=process_frames, daemon=True)
    thread.start()
    
    return True


def process_frames():
    global current_frame, is_running
    
    while is_running:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        # Process frame
        processed = proctor.process_frame(frame)
        
        with frame_lock:
            current_frame = processed
        
        time.sleep(0.03)  # ~30 FPS


def generate_frames():
    while True:
        with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           buffer.tobytes() + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    if proctor:
        return jsonify(proctor.get_stats())
    return jsonify({})


@app.route('/api/reset', methods=['POST'])
def reset_session():
    if proctor:
        proctor.start_session()
    return jsonify({'status': 'ok'})


def shutdown():
    global is_running
    is_running = False
    if camera:
        camera.release()


if __name__ == '__main__':
    print("=" * 50)
    print("AI Proctoring System - Web Interface")
    print("=" * 50)
    
    if init_system():
        print("\nServer starting...")
        print("Open http://localhost:5000 in your browser")
        print("Press Ctrl+C to stop\n")
        
        try:
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        finally:
            shutdown()
    else:
        print("Failed to initialize. Check your webcam.")
