# ============================================================================
# ar_whiteboard.py - draw in mid-air: the webcam watches your hand (MediaPipe
# hand tracking) and your fingertip becomes a pen on a virtual whiteboard.
# ============================================================================
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

MODEL_PATH = "hand_landmarker.task"

# Pseudo-enum for landmarks
class Lm:
    INDEX_FINGER_TIP = 8
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_PIP = 6
    MIDDLE_FINGER_TIP = 12
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_PIP = 10

def main():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: hand_landmarker.task not found! Please run hand_magic.py first to download it.")
        return

    # Initialize MediaPipe Tasks HandLandmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,         # Track one hand for drawing
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Colors in BGR format
    COLORS = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (0, 0, 0)       # Eraser 
    ]
    
    color_idx = 0
    brush_thickness = 10
    eraser_thickness = 50

    canvas = None
    prev_x, prev_y = 0, 0

    cap = cv2.VideoCapture(0)

    print("====================================")
    print("       AR Whiteboard Started!       ")
    print("====================================")
    print("Controls:")
    print(" - ONE Finger (Index): Draw")
    print(" - TWO Fingers (Index + Middle): Hover / Select tool from top menu")
    print(" - Q key: Quit program")
    print("====================================")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the frame horizontally for a natural selfie-view
        frame = cv2.flip(frame, 1)
        
        h, w, c = frame.shape
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # UI Overlay - Top Menu Bar
        # Red
        cv2.rectangle(frame, (20, 10), (120, 60), COLORS[0], -1)
        cv2.putText(frame, "Red", (45, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        # Green
        cv2.rectangle(frame, (140, 10), (240, 60), COLORS[1], -1)
        cv2.putText(frame, "Green", (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        # Blue
        cv2.rectangle(frame, (260, 10), (360, 60), COLORS[2], -1)
        cv2.putText(frame, "Blue", (290, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        # Yellow
        cv2.rectangle(frame, (380, 10), (480, 60), COLORS[3], -1)
        cv2.putText(frame, "Yellow", (395, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        # Eraser
        cv2.rectangle(frame, (500, 10), (600, 60), (255,255,255), -1)
        cv2.putText(frame, "Eraser", (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        # Clear
        cv2.rectangle(frame, (620, 10), (720, 60), (150, 150, 150), -1)
        cv2.putText(frame, "Clear", (645, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        
        # Process Hand Landmarks
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)
        
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            
            x1, y1 = int(landmarks[Lm.INDEX_FINGER_TIP].x * w), int(landmarks[Lm.INDEX_FINGER_TIP].y * h)
            x2, y2 = int(landmarks[Lm.MIDDLE_FINGER_TIP].x * w), int(landmarks[Lm.MIDDLE_FINGER_TIP].y * h)
            
            # Determine which fingers are up
            fingers = []
            # Index
            fingers.append(1 if landmarks[Lm.INDEX_FINGER_TIP].y < landmarks[Lm.INDEX_FINGER_PIP].y else 0)
            # Middle
            fingers.append(1 if landmarks[Lm.MIDDLE_FINGER_TIP].y < landmarks[Lm.MIDDLE_FINGER_PIP].y else 0)
            
            # Draw Hand Landmarks for visual feedback
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            
            # Hover / Selection Mode: Index AND Middle fingers are up
            if fingers[0] == 1 and fingers[1] == 1:
                prev_x, prev_y = 0, 0 # Reset drawing point
                # Draw hover indicator
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                
                # Menu Selection Logic
                if y1 < 70:
                    if 20 < x1 < 120: 
                        color_idx = 0
                    elif 140 < x1 < 240: 
                        color_idx = 1
                    elif 260 < x1 < 360: 
                        color_idx = 2
                    elif 380 < x1 < 480: 
                        color_idx = 3
                    elif 500 < x1 < 600: 
                        color_idx = 4 # Eraser
                    elif 620 < x1 < 720: 
                        canvas = np.zeros((h, w, 3), dtype=np.uint8) # Clear Canvas
                        
            # Drawing Mode: ONLY Index finger is up
            elif fingers[0] == 1 and fingers[1] == 0:
                # Indicator
                draw_color = COLORS[color_idx] if color_idx != 4 else (255,255,255)
                cv2.circle(frame, (x1, y1), 15, draw_color, cv2.FILLED)
                
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1
                    
                if color_idx == 4: # Eraser mode (Draw black on canvas)
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), (0, 0, 0), eraser_thickness)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), COLORS[color_idx], brush_thickness)
                    
                prev_x, prev_y = x1, y1
            else:
                # Reset if no drawing fingers are up
                prev_x, prev_y = 0, 0

        # Combine Frame and Canvas
        # Gray scale canvas out
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        # Create inverse mask where canvas has drawings
        _, img_inv = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        
        # Black out the drawing region in original frame
        frame = cv2.bitwise_and(frame, img_inv)
        # Add the drawing canvas on top of it
        frame = cv2.bitwise_or(frame, canvas)

        # Highlight currently selected tool
        if color_idx == 0:
            cv2.rectangle(frame, (20, 10), (120, 60), (255, 255, 255), 3)
        elif color_idx == 1:
            cv2.rectangle(frame, (140, 10), (240, 60), (255, 255, 255), 3)
        elif color_idx == 2:
            cv2.rectangle(frame, (260, 10), (360, 60), (255, 255, 255), 3)
        elif color_idx == 3:
            cv2.rectangle(frame, (380, 10), (480, 60), (255, 255, 255), 3)
        elif color_idx == 4:
            cv2.rectangle(frame, (500, 10), (600, 60), (0, 0, 0), 3)

        cv2.imshow("AR Whiteboard", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
