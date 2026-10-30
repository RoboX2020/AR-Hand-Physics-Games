# ============================================================================
# magic_slicer.py - Fruit-Ninja-style game: glowing orbs drift across the
# webcam view and you slice them with a swipe of your finger; sliced orbs
# burst into particles. Pure fun demo of hand tracking + simple physics.
# ============================================================================
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import random
import time
from collections import deque

MODEL_PATH = "hand_landmarker.task"

# Physics parameters
GRAVITY = 0.6
ORB_SPAWN_RATE = 0.03
MAX_LIVES = 3

# # One shard of a sliced orb.
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-8, 8)
        self.vy = random.uniform(-8, 8)
        self.life = 1.0
        self.color = color
        self.size = random.uniform(3, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += GRAVITY * 0.5
        self.life -= 0.03
        self.size *= 0.95

# # One half of a sliced orb: keeps flying with its own velocity and spin.
class OrbPiece:
    def __init__(self, x, y, vx, vy, color, radius):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = radius
        self.angle = 0
        self.spin = random.uniform(-10, 10)
        self.life = 1.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += GRAVITY
        self.angle += self.spin
        self.life -= 0.02

    def draw(self, img):
        if self.life > 0:
            c = tuple(int(ch * self.life) for ch in self.color)
            cv2.ellipse(img, (int(self.x), int(self.y)), (int(self.radius*0.8), int(self.radius*0.8)), 
                        int(self.angle), 0, 180, c, cv2.FILLED, cv2.LINE_AA)

# # A floating glowing orb drifting across the screen, waiting to be sliced.
class Orb:
    def __init__(self, w, h):
        self.x = random.randint(200, w - 200)
        self.y = h + 50
        # Arc towards center
        target_x = w / 2 + random.uniform(-300, 300)
        # Throw high
        self.vy = -random.uniform(22, 28)
        
        time_to_peak = -self.vy / GRAVITY
        self.vx = (target_x - self.x) / max(time_to_peak, 0.1)
        
        self.radius = random.randint(30, 60)
        self.color = random.choice([
            (0, 50, 255),    # Red/Orange
            (50, 255, 50),   # Green
            (255, 50, 255),  # Purple
            (50, 200, 255),  # Yellow
            (255, 200, 100)  # Cyan
        ])
        self.sliced = False
        self.missed = False

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += GRAVITY

    def draw(self, img):
        if not self.sliced:
            # Glow
            cv2.circle(img, (int(self.x), int(self.y)), self.radius + 10, tuple(int(c*0.4) for c in self.color), -1)
            cv2.circle(img, (int(self.x), int(self.y)), self.radius, self.color, -1)
            cv2.circle(img, (int(self.x - self.radius*0.3), int(self.y - self.radius*0.3)), int(self.radius*0.2), (255, 255, 255), -1)

# # How far a point is from a line - used to test whether your finger-swipe
# # cut through an orb.
def point_to_line_dist(px, py, lx1, ly1, lx2, ly2):
    line_mag = math.hypot(lx2 - lx1, ly2 - ly1)
    if line_mag == 0:
        return math.hypot(px - lx1, py - ly1)
    u = ((px - lx1) * (lx2 - lx1) + (py - ly1) * (ly2 - ly1)) / (line_mag ** 2)
    if u < 0:
        return math.hypot(px - lx1, py - ly1)
    elif u > 1:
        return math.hypot(px - lx2, py - ly2)
    else:
        ix = lx1 + u * (lx2 - lx1)
        iy = ly1 + u * (ly2 - ly1)
        return math.hypot(px - ix, py - iy)

def main():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found. Please run hand_magic.py to download it.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = "START"
    score = 0
    lives = MAX_LIVES
    
    orbs = []
    particles = []
    pieces = []
    
    # Store history for the blade
    blade_history = deque(maxlen=8)

    print("====================================")
    print("      AR Magic Slicer!              ")
    print("====================================")
    print("Controls:")
    print(" - Slice the flying orbs with your Index Finger!")
    print(" - Don't let them hit the floor. 3 Lives.")
    print(" - Press 'R' to Start / Restart")
    print(" - Press 'Q' to Quit")
    print("====================================")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # Darken bg for neon pop
        frame = cv2.convertScaleAbs(frame, alpha=0.5, beta=10)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            lm = results.hand_landmarks[0]
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)
            
            blade_history.append((ix, iy))
            
            # Draw glowing blade
            pts = list(blade_history)
            for i in range(1, len(pts)):
                cv2.line(overlay, pts[i-1], pts[i], (0, 255, 255), 10 + i*3, cv2.LINE_AA)
                cv2.line(overlay, pts[i-1], pts[i], (255, 255, 255), 3 + i, cv2.LINE_AA)
                
            # Slashing logic
            if state == "PLAYING" and len(pts) >= 2:
                for orb in orbs:
                    if not orb.sliced:
                        # Check last segments of blade 
                        for i in range(max(1, len(pts)-3), len(pts)):
                            lx1, ly1 = pts[i-1]
                            lx2, ly2 = pts[i]
                            
                            dist = point_to_line_dist(orb.x, orb.y, lx1, ly1, lx2, ly2)
                            stroke_speed = math.hypot(lx2 - lx1, ly2 - ly1)
                            
                            # Valid slash requires fast movement (stroke_speed > 15) and intersection (dist < radius)
                            if dist < orb.radius and stroke_speed > 15:
                                orb.sliced = True
                                score += 10
                                
                                # Burst particles
                                for _ in range(30):
                                    particles.append(Particle(orb.x, orb.y, orb.color))
                                
                                # Sliced halves
                                slash_dx = lx2 - lx1
                                slash_dy = ly2 - ly1
                                mag = max(1, math.hypot(slash_dx, slash_dy))
                                nx, ny = slash_dy / mag, -slash_dx / mag
                                
                                pieces.append(OrbPiece(orb.x, orb.y, orb.vx + nx*10, orb.vy + ny*10, orb.color, orb.radius))
                                pieces.append(OrbPiece(orb.x, orb.y, orb.vx - nx*10, orb.vy - ny*10, orb.color, orb.radius))
                                break
        else:
            blade_history.clear()

        # Update Game Objects
        if state == "PLAYING":
            difficulty = 1.0 + (score * 0.002)
            
            if random.random() < ORB_SPAWN_RATE * difficulty:
                orbs.append(Orb(w, h))
                
            for orb in orbs[:]:
                orb.update()
                orb.draw(overlay)
                
                # Missed
                if orb.y > h + 100 and not orb.sliced and not orb.missed:
                    orb.missed = True
                    lives -= 1
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1) # Red flash
                    
                if orb.y > h + 200 or orb.sliced:
                    orbs.remove(orb)
                    
            if lives <= 0:
                state = "GAME OVER"

        # Effects
        for p in particles: p.update()
        for p in pieces: p.update()
        
        particles = [p for p in particles if p.life > 0]
        pieces = [p for p in pieces if p.life > 0]
        
        for p in pieces: p.draw(overlay)
        for p in particles:
            c = tuple(int(x * p.life) for x in p.color)
            cv2.circle(overlay, (int(p.x), int(p.y)), max(1, int(p.size)), c, -1)

        # Draw UI
        frame = cv2.addWeighted(overlay, 0.9, frame, 1.0, 0)
        
        # HUD Panel
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.putText(frame, f"SCORE: {score}", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        lives_text = ""
        for i in range(MAX_LIVES):
            lives_text += " O " if i < lives else " X "
        cv2.putText(frame, "LIVES:", (w - 300, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        cv2.putText(frame, lives_text, (w - 180, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        if state == "START":
            cv2.putText(frame, "AR MAGIC SLICER", (w//2 - 250, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
            cv2.putText(frame, "Press 'R' to Start", (w//2 - 150, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif state == "GAME OVER":
            cv2.putText(frame, "GAME OVER", (w//2 - 200, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            cv2.putText(frame, "Press 'R' to Restart", (w//2 - 150, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("AR Magic Slicer", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            state = "PLAYING"
            score = 0
            lives = MAX_LIVES
            orbs.clear()
            particles.clear()
            pieces.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
