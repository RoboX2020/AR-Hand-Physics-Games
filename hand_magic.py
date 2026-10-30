# PLAIN ENGLISH: wave your hands to conjure particle magic on the webcam
# view (sparkles, trails, bursts that follow your fingers).
"""
Hand Magic AR Effects
=====================
Real-time AR magic effects mapped to both hands using MediaPipe Tasks API + OpenCV.
Press 1-8 to toggle effects, Q to quit.
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
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_TIP = 20

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CAMERA_INDEX = 0
WINDOW_NAME = "✨ Hand Magic AR ✨"
MAX_TRAIL_LENGTH = 40
MAX_PARTICLES = 300
ORB_COUNT = 5

# Color palettes (BGR)
MAGIC_COLORS = [
    (255, 100, 50),    # Electric blue
    (200, 50, 255),    # Purple
    (50, 255, 200),    # Cyan-green
    (100, 200, 255),   # Gold
    (255, 50, 150),    # Pink
    (50, 255, 100),    # Green
    (255, 180, 50),    # Sky blue
]

LIGHTNING_COLORS = [
    (255, 200, 150),   # Light blue
    (255, 150, 255),   # Light purple
    (200, 255, 255),   # Light yellow
]

AURA_COLORS = [
    (200, 50, 255),    # Purple
    (255, 100, 50),    # Blue
    (50, 200, 255),    # Orange-gold
]


# ─────────────────────────────────────────────
# Systems
# ─────────────────────────────────────────────
class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size', 'gravity']

    def __init__(self, x, y, vx, vy, life, color, size, gravity=0.15):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size
        self.gravity = gravity

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.vx *= 0.98
        self.life -= 1

    def is_alive(self):
        return self.life > 0


class TrailSystem:
    def __init__(self, max_length=MAX_TRAIL_LENGTH):
        self.trails = {}
        self.max_length = max_length

    def add_point(self, hand_idx, finger_idx, x, y):
        key = (hand_idx, finger_idx)
        if key not in self.trails:
            self.trails[key] = deque(maxlen=self.max_length)
        self.trails[key].append((x, y, time.time()))

    def get_trail(self, hand_idx, finger_idx):
        return self.trails.get((hand_idx, finger_idx), deque())

    def clear_old(self, max_age=0.8):
        now = time.time()
        for key in list(self.trails.keys()):
            while self.trails[key] and (now - self.trails[key][0][2]) > max_age:
                self.trails[key].popleft()
            if not self.trails[key]:
                del self.trails[key]


class InteractiveCube:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.x = w // 2
        self.y = 100
        self.z = 100
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.size = 60
        self.is_grabbed = False
        self.angle_x = 0.0
        self.angle_y = 0.0

    def update(self):
        if not self.is_grabbed:
            self.vy += 1.5  # gravity
            self.x += self.vx
            self.y += self.vy
            self.z += self.vz
            
            # Floor bounce
            if self.y > self.h - self.size:
                self.y = self.h - self.size
                self.vy = -self.vy * 0.7
                self.vx *= 0.9
                self.vz *= 0.9
            
            # Wall bounce
            if self.x < self.size:
                self.x = self.size
                self.vx = -self.vx * 0.8
            elif self.x > self.w - self.size:
                self.x = self.w - self.size
                self.vx = -self.vx * 0.8

        self.angle_x += 0.05 if not self.is_grabbed else 0.01
        self.angle_y += 0.03 if not self.is_grabbed else 0.01

    def draw(self, overlay):
        s = self.size
        # base vertices before rotation
        verts = []
        for ix in [-1, 1]:
            for iy in [-1, 1]:
                for iz in [-1, 1]:
                    verts.append(np.array([ix * s, iy * s, iz * s]))
        
        # rotation matrices
        rx = np.array([[1, 0, 0], [0, math.cos(self.angle_x), -math.sin(self.angle_x)], [0, math.sin(self.angle_x), math.cos(self.angle_x)]])
        ry = np.array([[math.cos(self.angle_y), 0, math.sin(self.angle_y)], [0, 1, 0], [-math.sin(self.angle_y), 0, math.cos(self.angle_y)]])
        
        edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
        
        f = 400.0  # focal length
        projected = []
        for v in verts:
            v_rot = ry.dot(rx.dot(v))
            px_3d = self.x + v_rot[0]
            py_3d = self.y + v_rot[1]
            pz_3d = self.z + v_rot[2]
            
            scale = f / (f + pz_3d) if (f + pz_3d) > 0 else 0.01
            px_2d = int((px_3d - self.w/2) * scale + self.w/2)
            py_2d = int((py_3d - self.h/2) * scale + self.h/2)
            projected.append((px_2d, py_2d))
            
        color = (50, 255, 100) if self.is_grabbed else (255, 100, 50)
        
        # Draw faces optionally, but wireframe is cooler with glow
        for e in edges:
            pt1 = projected[e[0]]
            pt2 = projected[e[1]]
            cv2.line(overlay, pt1, pt2, tuple(int(c * 0.5) for c in color), 6, cv2.LINE_AA)
            cv2.line(overlay, pt1, pt2, color, 2, cv2.LINE_AA)
        
        for p in projected:
            cv2.circle(overlay, p, 4, (255, 255, 255), -1)


# ─────────────────────────────────────────────
# Effect Renderers
# ─────────────────────────────────────────────

def draw_energy_orbs(overlay, palm_center, hand_idx, t):
    cx, cy = palm_center
    for i in range(ORB_COUNT):
        angle = t * (2.0 + i * 0.3) + (i * 2 * math.pi / ORB_COUNT) + hand_idx * math.pi
        radius = 50 + 15 * math.sin(t * 3 + i)
        ox = int(cx + radius * math.cos(angle))
        oy = int(cy + radius * math.sin(angle))

        color_idx = (i + hand_idx * 3) % len(MAGIC_COLORS)
        color = MAGIC_COLORS[color_idx]

        size = int(8 + 4 * math.sin(t * 5 + i * 1.5))
        cv2.circle(overlay, (ox, oy), size + 12, color, -1)
        cv2.circle(overlay, (ox, oy), size + 6, tuple(min(255, c + 60) for c in color), -1)
        cv2.circle(overlay, (ox, oy), size, (255, 255, 255), -1)


def draw_magic_trails(overlay, trail_system, hand_idx):
    fingertip_indices = [Lm.INDEX_FINGER_TIP, Lm.MIDDLE_FINGER_TIP, Lm.RING_FINGER_TIP, Lm.PINKY_TIP, Lm.THUMB_TIP]
    for fi, finger_idx in enumerate(fingertip_indices):
        trail = trail_system.get_trail(hand_idx, finger_idx)
        if len(trail) < 2: continue

        base_color = MAGIC_COLORS[(fi + hand_idx * 2) % len(MAGIC_COLORS)]
        points = list(trail)

        for j in range(1, len(points)):
            alpha = j / len(points)
            color = tuple(int(c * alpha) for c in base_color)
            pt1 = (int(points[j - 1][0]), int(points[j - 1][1]))
            pt2 = (int(points[j][0]), int(points[j][1]))
            cv2.line(overlay, pt1, pt2, color, int(1 + alpha * 6), cv2.LINE_AA)

        last = points[-1]
        cv2.circle(overlay, (int(last[0]), int(last[1])), 5, (255, 255, 255), -1)


def draw_lightning_arcs(overlay, landmarks, w, h, hand_idx, t):
    palm = landmarks[Lm.WRIST]
    palm_x, palm_y = int(palm.x * w), int(palm.y * h)

    fingertips = [Lm.INDEX_FINGER_TIP, Lm.MIDDLE_FINGER_TIP, Lm.RING_FINGER_TIP, Lm.PINKY_TIP]
    for fi, tip_idx in enumerate(fingertips):
        tip = landmarks[tip_idx]
        tx, ty = int(tip.x * w), int(tip.y * h)

        segments = 8
        points = []
        for s in range(segments + 1):
            frac = s / segments
            bx = palm_x + (tx - palm_x) * frac
            by = palm_y + (ty - palm_y) * frac
            if 0 < s < segments:
                jitter = 12 * (1 - abs(frac - 0.5) * 2)
                bx += random.uniform(-jitter, jitter)
                by += random.uniform(-jitter, jitter)
            points.append((int(bx), int(by)))

        color = LIGHTNING_COLORS[(fi + hand_idx) % len(LIGHTNING_COLORS)]
        for j in range(1, len(points)):
            cv2.line(overlay, points[j - 1], points[j], color, 2, cv2.LINE_AA)
            cv2.line(overlay, points[j - 1], points[j], (255, 255, 255), 1, cv2.LINE_AA)


def draw_particle_burst(overlay, particles):
    for p in particles:
        if not p.is_alive(): continue
        alpha = p.life / p.max_life
        size = max(1, int(p.size * alpha))
        color = tuple(int(c * alpha) for c in p.color)
        cv2.circle(overlay, (int(p.x), int(p.y)), size + 3, tuple(int(c * 0.4) for c in color), -1)
        cv2.circle(overlay, (int(p.x), int(p.y)), size, color, -1)


def spawn_particles(particles, cx, cy, count=20):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        life = random.randint(20, 50)
        color = random.choice(MAGIC_COLORS)
        size = random.randint(2, 5)
        particles.append(Particle(cx, cy, vx, vy, life, color, size, gravity=0.08))


def draw_energy_beam(overlay, palm1, palm2, t):
    x1, y1 = palm1
    x2, y2 = palm2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if dist > 400: return

    segments = 20
    intensity = max(0.2, 1.0 - dist / 400.0)

    for layer in range(3):
        points = []
        for s in range(segments + 1):
            frac = s / segments
            bx = x1 + (x2 - x1) * frac
            by = y1 + (y2 - y1) * frac
            dx, dy = x2 - x1, y2 - y1
            length = max(1, math.sqrt(dx * dx + dy * dy))
            nx, ny = -dy / length, dx / length

            wave = math.sin(frac * math.pi * 4 + t * 8 + layer * 2) * (10 + layer * 5)
            wave *= math.sin(frac * math.pi)

            bx += nx * wave
            by += ny * wave
            points.append((int(bx), int(by)))

        color = [(255, 100, 50), (200, 50, 255), (50, 255, 200)][layer % 3]
        color = tuple(int(c * intensity) for c in color)
        for j in range(1, len(points)):
            cv2.line(overlay, points[j - 1], points[j], color, 3 - layer, cv2.LINE_AA)

    for i in range(5):
        frac = (i + 1) / 6
        nx, ny = int(x1 + (x2 - x1) * frac), int(y1 + (y2 - y1) * frac)
        pulse = int(4 + 3 * math.sin(t * 6 + i * 1.2))
        color = tuple(int(c * intensity) for c in MAGIC_COLORS[i % len(MAGIC_COLORS)])
        cv2.circle(overlay, (nx, ny), pulse + 5, color, -1)
        cv2.circle(overlay, (nx, ny), pulse, (255, 255, 255), -1)


def draw_aura_glow(overlay, landmarks, w, h, hand_idx, t):
    hand_points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    if len(hand_points) < 5: return
    hull = cv2.convexHull(np.array(hand_points))
    base_color = AURA_COLORS[(int(t * 0.5) + hand_idx) % len(AURA_COLORS)]

    for layer in range(4):
        expansion = 10 + layer * 12 + int(5 * math.sin(t * 3 + layer))
        alpha_factor = 0.3 - layer * 0.06
        M = cv2.moments(hull)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        expanded = []
        for pt in hull:
            px, py = pt[0]
            dx, dy = px - cx, py - cy
            dist = max(1, math.sqrt(dx * dx + dy * dy))
            scale = (dist + expansion) / dist
            expanded.append([int(cx + dx * scale), int(cy + dy * scale)])

        color = tuple(int(c * alpha_factor) for c in base_color)
        cv2.fillConvexPoly(overlay, np.array(expanded, dtype=np.int32), color, cv2.LINE_AA)


def draw_wizard_wand(overlay, landmarks, w, h, hand_idx, past_tips, spells):
    wrist = landmarks[Lm.WRIST]
    index_mcp = landmarks[Lm.INDEX_FINGER_MCP]
    
    wx, wy = int(wrist.x * w), int(wrist.y * h)
    ix, iy = int(index_mcp.x * w), int(index_mcp.y * h)
    
    dx, dy = ix - wx, iy - wy
    length = math.sqrt(dx*dx + dy*dy)
    if length < 1: return
    
    wand_length = 200
    nx, ny = dx/length, dy/length
    
    tip_x = int(ix + nx * wand_length)
    tip_y = int(iy + ny * wand_length)
    
    base_color = (30, 70, 110) # Wood dark brown
    
    # Wand thickness
    perp_x, perp_y = -ny, nx
    thickness = 12
    pts = np.array([
        [wx + perp_x * thickness, wy + perp_y * thickness],
        [wx - perp_x * thickness, wy - perp_y * thickness],
        [tip_x - perp_x * 4, tip_y - perp_y * 4],
        [tip_x + perp_x * 4, tip_y + perp_y * 4]
    ], np.int32)
    cv2.fillConvexPoly(overlay, pts, base_color, cv2.LINE_AA)
    
    # Glowing star tip
    cv2.circle(overlay, (tip_x, tip_y), 15, (255, 150, 50), -1)
    cv2.circle(overlay, (tip_x, tip_y), 7, (255, 255, 255), -1)
    
    # Flick detection
    t = time.time()
    past_tips.append((tip_x, tip_y, t))
    while len(past_tips) > 6:
        past_tips.pop(0)
        
    if len(past_tips) >= 4:
        old_x, old_y, old_t = past_tips[0]
        dt = t - old_t
        if dt > 0.01:
            vx = (tip_x - old_x) / dt
            vy = (tip_y - old_y) / dt
            speed = math.sqrt(vx*vx + vy*vy)
            if speed > 1800 and len(spells) < 15:  # Flick!
                spells.append({
                    'x': tip_x, 'y': tip_y,
                    'vx': vx * 0.015, 'vy': vy * 0.015,
                    'life': 70,
                    'color': random.choice(MAGIC_COLORS)
                })
                past_tips.clear()


def update_and_draw_spells(overlay, particles, spells, w, h):
    for spell in spells[:]:
        spell['life'] -= 1
        spell['x'] += spell['vx']
        spell['y'] += spell['vy']
        spell['vy'] += 0.2  # Gravity for spells too
        
        # Bouncing
        if spell['x'] < 20 or spell['x'] > w - 20: spell['vx'] *= -0.8
        if spell['y'] < 20 or spell['y'] > h - 20: spell['vy'] *= -0.8
        
        # Leave a trail using particles
        if spell['life'] > 0 and random.random() < 0.6:
            particles.append(Particle(spell['x'], spell['y'], -spell['vx']*0.1, -spell['vy']*0.1, 15, spell['color'], 3))
        
        if spell['life'] <= 0:
            # Explosion
            spawn_particles(particles, spell['x'], spell['y'], 10)
            spells.remove(spell)
            continue
            
        color = spell['color']
        cv2.circle(overlay, (int(spell['x']), int(spell['y'])), 18, tuple(int(c*0.5) for c in color), -1)
        cv2.circle(overlay, (int(spell['x']), int(spell['y'])), 8, (255, 255, 255), -1)


# ─────────────────────────────────────────────
# Hand state tracking for gesture detection
# ─────────────────────────────────────────────
class HandState:
    def __init__(self):
        self.prev_openness = 0.0
        self.was_open = False
        self.wand_past_tips = []

    def get_openness(self, landmarks, w, h):
        wrist = landmarks[Lm.WRIST]
        tips = [landmarks[Lm.INDEX_FINGER_TIP], landmarks[Lm.MIDDLE_FINGER_TIP], landmarks[Lm.RING_FINGER_TIP], landmarks[Lm.PINKY_TIP]]
        mcps = [landmarks[Lm.INDEX_FINGER_MCP], landmarks[Lm.MIDDLE_FINGER_MCP], landmarks[Lm.RING_FINGER_MCP], landmarks[Lm.PINKY_MCP]]

        total = 0
        for tip, mcp in zip(tips, mcps):
            tip_dist = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2)
            mcp_dist = math.sqrt((mcp.x - wrist.x) ** 2 + (mcp.y - wrist.y) ** 2)
            ratio = tip_dist / max(mcp_dist, 0.001)
            total += min(ratio / 2.0, 1.0)
        return total / 4.0

    def detect_burst(self, landmarks, w, h):
        openness = self.get_openness(landmarks, w, h)
        is_open = openness > 0.6
        burst = is_open and not self.was_open
        self.was_open = is_open
        self.prev_openness = openness
        return burst


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  ✨ Hand Magic AR Effects ✨")
    print("=" * 50)
    print("Controls:")
    print("  1 - Toggle Energy Orbs")
    print("  2 - Toggle Magic Trails")
    print("  3 - Toggle Lightning Arcs")
    print("  4 - Toggle Particle Bursts")
    print("  5 - Toggle Energy Beam")
    print("  6 - Toggle Aura Glow")
    print("  7 - Toggle Wizard Wand (Effect)")
    print("  8 - Toggle 3D Cube (Pinch to Grab)")
    print("  Q - Quit")
    print("=" * 50)

    download_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

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

    # Systems
    trail_system = TrailSystem()
    particles = []
    spells = []
    
    # Resolution placeholder for initialization
    ret, frame = cap.read()
    test_h, test_w = 720, 1280
    if ret: test_h, test_w = frame.shape[:2]
    
    cube = InteractiveCube(test_w, test_h)
    hand_states = [HandState(), HandState()]

    # Toggles
    effects_on = {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True}
    effect_names = {
        1: "Orbs", 2: "Trails", 3: "Lightng", 
        4: "Burst", 5: "Beam", 6: "Aura", 
        7: "Wand", 8: "Cube"
    }

    start_time = time.time()
    fps_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cube.w, cube.h = w, h
        t = time.time() - start_time

        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        overlay = np.zeros_like(frame)
        palm_centers = []

        is_cube_grabbed_this_frame = False

        if results.hand_landmarks:
            for hand_idx, landmarks in enumerate(results.hand_landmarks):
                # Palm center
                palm_lms = [Lm.WRIST, Lm.INDEX_FINGER_MCP, Lm.MIDDLE_FINGER_MCP, Lm.RING_FINGER_MCP, Lm.PINKY_MCP]
                palm_x = int(np.mean([landmarks[i].x for i in palm_lms]) * w)
                palm_y = int(np.mean([landmarks[i].y for i in palm_lms]) * h)
                palm_center = (palm_x, palm_y)
                palm_centers.append(palm_center)

                # Interactions
                if hand_idx < len(hand_states):
                    if hand_states[hand_idx].detect_burst(landmarks, w, h) and effects_on[4]:
                        spawn_particles(particles, palm_x, palm_y, count=30)

                # Interactive Cube Grabbing
                if effects_on[8]:
                    tx, ty = landmarks[Lm.THUMB_TIP].x * w, landmarks[Lm.THUMB_TIP].y * h
                    ix, iy = landmarks[Lm.INDEX_FINGER_TIP].x * w, landmarks[Lm.INDEX_FINGER_TIP].y * h
                    pinch_dist = math.sqrt((tx-ix)**2 + (ty-iy)**2)
                    mid_x, mid_y = (tx+ix)/2, (ty+iy)/2
                    
                    cube_dist = math.sqrt((cube.x - mid_x)**2 + (cube.y - mid_y)**2)
                    if pinch_dist < 45 and cube_dist < 80:
                        is_cube_grabbed_this_frame = True
                        cube.is_grabbed = True
                        cube.vx = (mid_x - cube.x) * 0.4
                        cube.vy = (mid_y - cube.y) * 0.4
                        cube.x, cube.y = mid_x, mid_y

                # Add trails
                for finger_idx in [Lm.INDEX_FINGER_TIP, Lm.MIDDLE_FINGER_TIP, Lm.RING_FINGER_TIP, Lm.PINKY_TIP, Lm.THUMB_TIP]:
                    trail_system.add_point(hand_idx, finger_idx, landmarks[finger_idx].x * w, landmarks[finger_idx].y * h)

                # Draws
                if effects_on[6]: draw_aura_glow(overlay, landmarks, w, h, hand_idx, t)
                if effects_on[1]: draw_energy_orbs(overlay, palm_center, hand_idx, t)
                if effects_on[3]: draw_lightning_arcs(overlay, landmarks, w, h, hand_idx, t)
                if effects_on[7]: draw_wizard_wand(overlay, landmarks, w, h, hand_idx, hand_states[hand_idx].wand_past_tips, spells)

            if effects_on[5] and len(palm_centers) >= 2:
                draw_energy_beam(overlay, palm_centers[0], palm_centers[1], t)

        if effects_on[8] and not is_cube_grabbed_this_frame:
            cube.is_grabbed = False

        if effects_on[2]:
            for hidx in range(2): draw_magic_trails(overlay, trail_system, hidx)
        trail_system.clear_old()

        # Update systems
        particles = [p for p in particles if p.is_alive()]
        for p in particles: p.update()
        if effects_on[4]: draw_particle_burst(overlay, particles)

        if effects_on[7]: update_and_draw_spells(overlay, particles, spells, w, h)
        if effects_on[8]: 
            cube.update()
            cube.draw(overlay)

        # Ambient particles
        if effects_on[4] and results.hand_landmarks:
            for palm_center in palm_centers:
                if random.random() < 0.3:
                    spawn_particles(particles, palm_center[0], palm_center[1], count=2)

        # Composite
        darkened = (frame * 0.7).astype(np.uint8)
        result = cv2.add(darkened, overlay)

        # HUD
        hud_bar = np.zeros((40, w, 3), dtype=np.uint8)
        hud_bar[:] = (30, 30, 30)
        result[0:40] = cv2.addWeighted(result[0:40], 0.5, hud_bar, 0.5, 0)
        cv2.putText(result, f"FPS: {fps}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        status_x = 100
        for key in sorted(effects_on.keys()):
            color = (50, 255, 50) if effects_on[key] else (100, 100, 100)
            text = f"{key}:{effect_names[key]}"
            cv2.putText(result, text, (status_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            status_x += 105

        n_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
        cv2.putText(result, f"Hands: {n_hands}", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'): break
        elif key in [ord(str(i)) for i in range(1, 9)]:
            toggle = int(chr(key))
            effects_on[toggle] = not effects_on[toggle]

    cap.release()
    cv2.destroyAllWindows()
    print("\nGoodbye! ✨")

if __name__ == "__main__":
    main()
