#open http://localhost:5000 in browser

import cv2
import time
import threading
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request

from proctor import ProctorSystem

app = Flask(__name__)

proctor = None
camera = None
is_running = False
frame_lock = threading.Lock()
current_frame = None
violation_log = []
session_start_time = None
settings = {
    'sensitivity': 'medium',
    'alert_sound': True,
    'auto_pause': False,
    'record_violations': True
}


def init_system():
    global proctor, camera, is_running, session_start_time
    
    proctor = ProctorSystem()
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("ERROR: Cannot open webcam!")
        return False
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    proctor.start_session()
    is_running = True
    session_start_time = datetime.now()
    
    thread = threading.Thread(target=process_frames, daemon=True)
    thread.start()
    
    return True


def process_frames():
    global current_frame, is_running, violation_log
    
    while is_running:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        processed = proctor.process_frame(frame)
        
        stats = proctor.get_stats()
        if stats.get('current_violation') and settings['record_violations']:
            violation_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': stats.get('current_violation'),
                'duration': stats.get('violation_duration', 0)
            })
            if len(violation_log) > 100:
                violation_log.pop(0)
        
        with frame_lock:
            current_frame = processed
        
        time.sleep(0.03)


def generate_frames():
    while True:
        with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
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
        stats = proctor.get_stats()
        stats['session_duration'] = str(datetime.now() - session_start_time).split('.')[0] if session_start_time else '00:00:00'
        stats['total_logged_violations'] = len(violation_log)
        return jsonify(stats)
    return jsonify({})


@app.route('/api/violations')
def get_violations():
    return jsonify(violation_log[-50:])


@app.route('/api/reset', methods=['POST'])
def reset_session():
    global violation_log, session_start_time
    if proctor:
        proctor.start_session()
        violation_log = []
        session_start_time = datetime.now()
    return jsonify({'status': 'ok'})


@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global settings
    if request.method == 'POST':
        data = request.get_json()
        settings.update(data)
        return jsonify({'status': 'ok', 'settings': settings})
    return jsonify(settings)


@app.route('/api/export')
def export_report():
    report = {
        'session_start': session_start_time.isoformat() if session_start_time else None,
        'session_duration': str(datetime.now() - session_start_time).split('.')[0] if session_start_time else None,
        'total_violations': len(violation_log),
        'violations': violation_log,
        'final_stats': proctor.get_stats() if proctor else {}
    }
    return jsonify(report)


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