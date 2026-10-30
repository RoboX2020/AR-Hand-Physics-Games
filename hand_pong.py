# PLAIN ENGLISH: classic Pong, but the paddles are your hands - move each
# hand up and down in front of the webcam to keep the ball in play.
"""
AR Hand Pong
============
A two-player AR ping-pong game using your hands as paddles.
Uses MediaPipe Hand Tracking and OpenCV.
Press 'R' to reset score, 'Q' to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
import random
import os
import urllib.request
from collections import deque

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Hand Landmarks pseudo-enum
class Lm:
    WRIST = 0
    INDEX_FINGER_MCP = 5
    MIDDLE_FINGER_MCP = 9
    RING_FINGER_MCP = 13
    PINKY_MCP = 17

# ─────────────────────────────────────────────
# Systems
# ─────────────────────────────────────────────
class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size', 'friction']

    def __init__(self, x, y, vx, vy, life, color, size, friction=0.92):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size
        self.friction = friction

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.friction
        self.vy *= self.friction
        self.life -= 1

    def is_alive(self):
        return self.life > 0

def spawn_explosion(particles, x, y, color, count=30, speed=10):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        v = random.uniform(speed * 0.2, speed)
        vx = math.cos(angle) * v
        vy = math.sin(angle) * v
        life = random.randint(15, 30)
        size = random.randint(2, 6)
        particles.append(Particle(x, y, vx, vy, life, color, size))

class Ball:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.base_speed = 12.0
        self.radius = 15
        self.color = (255, 255, 255)
        self.trail = deque(maxlen=15)
        self.reset()

    def increase_speed(self, amount):
        self.base_speed = min(self.base_speed + amount, 40.0)
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        if current_speed > 0:
            scale = (current_speed + amount) / current_speed
            self.vx *= scale
            self.vy *= scale

    def decrease_speed(self, amount):
        self.base_speed = max(self.base_speed - amount, 5.0)
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        if current_speed > 0:
            scale = max((current_speed - amount), 5.0) / current_speed
            self.vx *= scale
            self.vy *= scale

    def reset(self, direction=None):
        self.x = self.w / 2
        self.y = self.h / 2
        speed = self.base_speed
        
        # Decide direction
        if direction is None:
            direction = random.choice([1, -1])
            
        angle = random.uniform(-math.pi/6, math.pi/6) # Within 30 degrees of horizontal
        if direction == -1:
            angle += math.pi
            
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        
        self.trail.clear()

    def update(self, particles):
        self.trail.append((self.x, self.y))
        
        self.x += self.vx
        self.y += self.vy

        # Top and bottom wall bounces
        did_bounce = False
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
            did_bounce = True
        elif self.y + self.radius > self.h:
            self.y = self.h - self.radius
            self.vy = -self.vy
            did_bounce = True
            
        if did_bounce:
            spawn_explosion(particles, self.x, self.y, (150, 150, 150), count=10, speed=5)

    def draw(self, overlay):
        # Draw trail
        points = list(self.trail)
        for i in range(1, len(points)):
            alpha = i / len(points)
            color = tuple(int(c * alpha) for c in (100, 200, 255))
            thickness = int(self.radius * alpha)
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
        
        # Draw ball
        ix, iy = int(self.x), int(self.y)
        cv2.circle(overlay, (ix, iy), self.radius + 10, (100, 200, 255), -1)
        cv2.circle(overlay, (ix, iy), self.radius, (255, 255, 255), -1)

# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("      🏓 AR Hand Pong 🏓      ")
    print("=" * 50)
    print("Use your hands to hit the ball!")
    print("Left side = Player 1 (Blue)")
    print("Right side = Player 2 (Red)")
    print("Controls:")
    print("  + / = - Increase Speed")
    print("  -     - Decrease Speed")
    print("  R     - Reset Score")
    print("  Q     - Quit")
    print("=" * 50)

    download_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    # MediaPipe setup
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Game State
    ret, frame = cap.read()
    h, w = 720, 1280
    if ret: h, w = frame.shape[:2]

    ball = Ball(w, h)
    particles = []
    
    score_p1 = 0
    score_p2 = 0
    
    paddle_radius = 50

    CAMERA_INDEX = 0
    WINDOW_NAME = "🏓 AR Hand Pong 🏓"

    start_time = time.time()
    fps_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Mirror so left is left, right is right
        h, w = frame.shape[:2]
        ball.w, ball.h = w, h
        t = time.time() - start_time

        # FPS
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        overlay = np.zeros_like(frame)

        # 1. Update Game Physics
        ball.update(particles)

        # 2. Check collisions with paddles (hands)
        paddles = []
        if results.hand_landmarks:
            for landmarks in results.hand_landmarks:
                # Calculate palm center
                palm_lms = [Lm.WRIST, Lm.INDEX_FINGER_MCP, Lm.MIDDLE_FINGER_MCP, Lm.RING_FINGER_MCP, Lm.PINKY_MCP]
                px = np.mean([landmarks[i].x for i in palm_lms]) * w
                py = np.mean([landmarks[i].y for i in palm_lms]) * h
                
                # Determine player visually
                is_p1 = px < w / 2
                color = (255, 100, 50) if is_p1 else (50, 100, 255) # BGR: P1=Blue, P2=Red
                
                paddles.append({'x': px, 'y': py, 'color': color, 'is_p1': is_p1})

        # Process collisions
        for p in paddles:
            dx = ball.x - p['x']
            dy = ball.y - p['y']
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < ball.radius + paddle_radius:
                # Collision detected!
                spawn_explosion(particles, ball.x, ball.y, p['color'], count=20, speed=15)
                
                # Calculate normal
                if dist == 0:
                    nx, ny = 1, 0
                else:
                    nx, ny = dx/dist, dy/dist
                
                # Reflect velocity: v = v - 2(v.n)n
                dot = ball.vx * nx + ball.vy * ny
                
                # Only bounce if headed towards the paddle (dot < 0)
                if dot < 0:
                    ball.vx -= 2 * dot * nx
                    ball.vy -= 2 * dot * ny
                    
                    # Add a bit of speed on hit, cap at max
                    speed = math.sqrt(ball.vx**2 + ball.vy**2)
                    new_speed = min(speed * 1.05 + 1.0, 45.0)
                    ball.vx = (ball.vx / speed) * new_speed
                    ball.vy = (ball.vy / speed) * new_speed
                    
                    # Push ball out of deadzone
                    overlap = (ball.radius + paddle_radius) - dist
                    ball.x += nx * overlap
                    ball.y += ny * overlap

        # 3. Check for scoring
        if ball.x < 0:
            score_p2 += 1
            spawn_explosion(particles, 0, ball.y, (50, 100, 255), count=50, speed=20)
            ball.reset(1) # Serve to P2
        elif ball.x > w:
            score_p1 += 1
            spawn_explosion(particles, w, ball.y, (255, 100, 50), count=50, speed=20)
            ball.reset(-1) # Serve to P1

        # 4. Draw Effects
        # Draw Center Line
        for y in range(0, h, 40):
            cv2.line(overlay, (w//2, y), (w//2, y+20), (50, 50, 50), 4)

        # Draw Paddles
        for p in paddles:
            ix, iy = int(p['x']), int(p['y'])
            # Soft glow
            cv2.circle(overlay, (ix, iy), paddle_radius + 15, tuple(int(c*0.4) for c in p['color']), -1)
            cv2.circle(overlay, (ix, iy), paddle_radius, p['color'], -1)
            cv2.circle(overlay, (ix, iy), paddle_radius - 5, (255, 255, 255), 2)

        # Draw Ball
        ball.draw(overlay)

        # Draw particles
        particles = [p for p in particles if p.is_alive()]
        for p in particles:
            p.update()
            alpha = p.life / p.max_life
            sz = max(1, int(p.size * alpha))
            color = tuple(int(c * alpha) for c in p.color)
            cv2.circle(overlay, (int(p.x), int(p.y)), sz, color, -1)

        # Composite with darkened background
        darkened = (frame * 0.4).astype(np.uint8)
        result = cv2.add(darkened, overlay)

        # 5. UI / Scoreboard
        # Shadow for scores
        cv2.putText(result, f"{score_p1}", (w//4 - 50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(result, f"{score_p2}", (3*w//4 - 50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)
        # Colored scores (P1: Blue, P2: Red in BGR)
        cv2.putText(result, f"{score_p1}", (w//4 - 50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 50), 5, cv2.LINE_AA)
        cv2.putText(result, f"{score_p2}", (3*w//4 - 50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 100, 255), 5, cv2.LINE_AA)

        cv2.putText(result, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        current_speed = int(math.sqrt(ball.vx**2 + ball.vy**2))
        cv2.putText(result, f"Speed: {current_speed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 100), 1, cv2.LINE_AA)
        
        cv2.putText(result, "Press '+/-' to change speed, 'R' to reset score", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            score_p1 = 0
            score_p2 = 0
            ball.reset()
        elif key == ord('=') or key == ord('+'):
            ball.increase_speed(3.0)
        elif key == ord('-') or key == ord('_'):
            ball.decrease_speed(3.0)

    cap.release()
    cv2.destroyAllWindows()
    print("\nGoodbye! 🏓")

if __name__ == "__main__":
    main()
