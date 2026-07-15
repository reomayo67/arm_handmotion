import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
from collections import deque

# Global variable for landmarker result
landmarker_result = None

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global landmarker_result
    landmarker_result = result

# Download hand landmarker model if not exists
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    print(f"Downloading hand landmarker model from {model_url}...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded!")

# --- ATTEMPT GPU INITIALIZATION WITH CPU FALLBACK ---
def create_landmarker(delegate_type):
    base_options = python.BaseOptions(
        model_asset_path=model_path, 
        delegate=delegate_type
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.4, # Lowered to tolerate heavy blur
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        result_callback=result_callback
    )
    return vision.HandLandmarker.create_from_options(options)

try:
    print("Attempting to initialize MediaPipe with GPU...")
    hand_landmarker = create_landmarker(python.BaseOptions.Delegate.GPU)
    print("SUCCESS: GPU Acceleration Enabled!")
except Exception as e:
    print("NOTICE: GPU delegate failed. Falling back to CPU...")
    hand_landmarker = create_landmarker(python.BaseOptions.Delegate.CPU)
    print("SUCCESS: CPU Fallback Enabled.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
else:
    h, w = 480, 640

print(f"Webcam resolution: {w}x{h}")
print("Press 'q' to quit, 'c' to clear drawing, 't' to toggle trail, 's' to save")

trail_canvas = np.zeros((h, w, 3), dtype=np.uint8)
show_trail = True

draw_w, draw_h = 500, 500
draw_canvas = np.zeros((draw_h, draw_w, 3), dtype=np.uint8)
draw_trail = []

map_w, map_h = 500, 500
map_center_x, map_center_y = map_w // 2, map_h // 2

# Tracking state
ema_x, ema_y = map_center_x, map_center_y
fist_history = deque(maxlen=5)
fist = False

# Pen State
was_pen_down = False
prev_palm_map_x, prev_palm_map_y = map_center_x, map_center_y

# Motion Blur Grace Period
frames_missing = 0
MAX_DROPOUT_FRAMES = 15  # Increased to ~250ms tolerance for fast swipes

def analyze_hand_state(hand_landmarks):
    """
    Advanced vector-based hand analysis (THUMB REMOVED).
    """
    def get_vector(idx1, idx2):
        p1 = hand_landmarks[idx1]
        p2 = hand_landmarks[idx2]
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def get_finger_curl(mcp, pip, dip, tip):
        v_base = get_vector(mcp, pip)
        v_tip = get_vector(dip, tip)
        
        norm_base = np.linalg.norm(v_base)
        norm_tip = np.linalg.norm(v_tip)
        if norm_base == 0 or norm_tip == 0:
            return 1.0 
            
        v_base = v_base / norm_base
        v_tip = v_tip / norm_tip
        
        return np.dot(v_base, v_tip)

    fingers = {
        'Index': (5, 6, 7, 8),
        'Middle': (9, 10, 11, 12),
        'Ring': (13, 14, 15, 16),
        'Pinky': (17, 18, 19, 20)
    }
    
    curled_count = 0
    for name, indices in fingers.items():
        # THRESHOLD RAISED TO 0.4: Motion blur makes curled fingers look straighter.
        # This makes the detection much more forgiving during movement.
        if get_finger_curl(*indices) < 0.4: 
            curled_count += 1

    return (curled_count >= 3)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    hand_landmarker.detect_async(mp_image, timestamp_ms)
    
    # Initialize UI Canvases
    draw_display = draw_canvas.copy()
    
    map_canvas = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    for i, (trail_x, trail_y, _) in enumerate(draw_trail[-200:]):
        alpha = i / max(1, len(draw_trail[-200:]))
        color = (int(100 + 155 * alpha), int(255 * alpha), int(100 + 155 * alpha))
        cv2.circle(map_canvas, (trail_x, trail_y), 2, color, -1)
    
    palm_detected = False
    is_moving_fast = False
    
    if landmarker_result and landmarker_result.hand_landmarks:
        palm_detected = True
        frames_missing = 0 
        
        hand_landmarks = landmarker_result.hand_landmarks[0]
        
        # Draw basic hand skeleton
        connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection.start, connection.end
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            cv2.line(frame, (int(start.x * w), int(start.y * h)), 
                     (int(end.x * w), int(end.y * h)), (255, 100, 0), 2)
        
        tracker_point = hand_landmarks[9]
        palm_x_px, palm_y_px = int(tracker_point.x * w), int(tracker_point.y * h)
        cv2.circle(frame, (palm_x_px, palm_y_px), 8, (0, 255, 255), -1)
        
        # --- VELOCITY CALCULATION ---
        raw_map_x = tracker_point.x * draw_w
        raw_map_y = tracker_point.y * draw_h
        velocity = np.sqrt((raw_map_x - ema_x)**2 + (raw_map_y - ema_y)**2)
        
        # Determine if hand is moving fast (Threshold: 25 pixels per frame)
        is_moving_fast = velocity > 25.0
        
        # --- KINEMATIC STATE LOCKING ---
        if is_moving_fast and len(fist_history) > 0:
            # If moving fast, completely ignore the camera's distorted view of the fingers.
            # Just copy whatever state the hand was in before it started moving fast.
            fist_raw = fist_history[-1] 
            cv2.putText(frame, "SPEED LOCK ACTIVE", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
        else:
            # If moving slow enough for a clear picture, analyze the fingers normally
            fist_raw = analyze_hand_state(hand_landmarks)
            
        fist_history.append(fist_raw)
        fist = sum(fist_history) >= 3 
        
        # --- DYNAMIC SMOOTHING ---
        dynamic_alpha = np.clip(velocity / 40.0, 0.15, 0.9)
        ema_x = dynamic_alpha * raw_map_x + (1 - dynamic_alpha) * ema_x
        ema_y = dynamic_alpha * raw_map_y + (1 - dynamic_alpha) * ema_y
        
        palm_map_x = int(ema_x)
        palm_map_y = int(ema_y)
        
    else:
        frames_missing += 1
        if frames_missing > MAX_DROPOUT_FRAMES:
            fist_history.clear()
            fist = False
        else:
            pass # Inside dropout grace period

    # --- PEN DRAWING LOGIC ---
    if palm_detected and fist:
        if not was_pen_down:
            prev_palm_map_x, prev_palm_map_y = palm_map_x, palm_map_y
            
        cv2.line(draw_canvas, (prev_palm_map_x, prev_palm_map_y), 
                 (palm_map_x, palm_map_y), (255, 255, 255), 3)
        draw_trail.append((palm_map_x, palm_map_y, 1.0))
        if len(draw_trail) > 1000: draw_trail.pop(0)
            
        prev_palm_map_x, prev_palm_map_y = palm_map_x, palm_map_y
        was_pen_down = True
        
    elif not palm_detected and fist:
        pass # Bridge gap during dropout
        
    else:
        was_pen_down = False

    # --- RENDERING ---
    pen_status = was_pen_down or (fist and frames_missing <= MAX_DROPOUT_FRAMES)
    dot_color = (0, 255, 0) if pen_status else (0, 0, 255)
    
    cv2.putText(draw_display, f"Pen: {'DOWN' if pen_status else 'UP'}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color, 2)
    
    if palm_detected:
        cv2.circle(map_canvas, (palm_map_x, palm_map_y), 10, dot_color, 2)
        cv2.circle(draw_display, (palm_map_x, palm_map_y), 6, dot_color, -1)
        cv2.putText(frame, f"FIST: {fist}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, dot_color, 3)
    elif frames_missing <= MAX_DROPOUT_FRAMES and fist:
        cv2.putText(frame, "MOTION BLUR DETECTED - BRIDGING GAP", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    combined = cv2.addWeighted(frame, 0.7, trail_canvas, 0.3, 0) if show_trail else frame
    
    cv2.imshow('Hand Tracker - Webcam', combined)
    cv2.imshow('Hand Tracker - Position Map', map_canvas)
    cv2.imshow('Hand Tracker - Drawing Canvas', draw_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'):
        draw_canvas = np.zeros((draw_h, draw_w, 3), dtype=np.uint8)
        draw_trail.clear()
    elif key == ord('t'): show_trail = not show_trail

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()