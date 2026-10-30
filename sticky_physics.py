# ============================================================================
# sticky_physics.py - balls bounce off real objects: hold up anything of a
# bright color and the camera turns it into a solid paddle/wall that the
# simulated balls collide with (color detection -> physics obstacles).
# ============================================================================
import cv2
import numpy as np
import math
import time
import random

MAX_BALLS = 30  # Performance cap

# ─────────────────────────────────────────────
# Physics Ball
# ─────────────────────────────────────────────
class Ball:
    def __init__(self, x, y, radius=12):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = 0.0
        self.radius = radius
        self.color = random.choice([
            (50, 200, 255), (50, 255, 100), (255, 100, 255),
            (255, 200, 50), (100, 255, 255), (255, 50, 150),
        ])
        self.restitution = 0.72
        self.trail = []

    def update(self, gravity=0.5, dt=1.0):
        self.vy += gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.999
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 12:
            self.trail.pop(0)

    def alive(self, w, h):
        # Dies when it falls off bottom or sides — no floor bounce
        return -50 < self.x < w + 50 and self.y < h + 50

    def draw(self, img):
        for i in range(1, len(self.trail)):
            alpha = i / len(self.trail)
            c = tuple(int(ch * alpha * 0.5) for ch in self.color)
            cv2.line(img, self.trail[i-1], self.trail[i], c,
                     max(1, int(self.radius * 0.5 * alpha)), cv2.LINE_AA)
        cv2.circle(img, (int(self.x), int(self.y)), self.radius + 5,
                   tuple(int(c*0.3) for c in self.color), -1, cv2.LINE_AA)
        cv2.circle(img, (int(self.x), int(self.y)), self.radius,
                   self.color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(self.x)-3, int(self.y)-3),
                   max(1, self.radius//3), (255,255,255), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Collision
# ─────────────────────────────────────────────
def closest_point_on_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length_sq = dx*dx + dy*dy
    if length_sq == 0: return ax, ay
    t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / length_sq))
    return ax + t*dx, ay + t*dy

def collide_ball_segment(ball, ax, ay, bx, by):
    cx, cy = closest_point_on_segment(ball.x, ball.y, ax, ay, bx, by)
    dist = math.hypot(ball.x - cx, ball.y - cy)
    if dist < ball.radius and dist > 0.001:
        nx = (ball.x - cx) / dist
        ny = (ball.y - cy) / dist
        overlap = ball.radius - dist
        ball.x += nx * overlap * 1.05
        ball.y += ny * overlap * 1.05
        dot = ball.vx * nx + ball.vy * ny
        ball.vx -= 2 * dot * nx
        ball.vy -= 2 * dot * ny
        ball.vx *= ball.restitution
        ball.vy *= ball.restitution
        return True
    return False


# ─────────────────────────────────────────────
# Stable Color-Based Rectangle Detector
# ─────────────────────────────────────────────
# HSV ranges for common bright sticky note colors
COLOR_RANGES = [
    # Orange sticky notes
    {"lower": (5, 100, 100), "upper": (25, 255, 255), "name": "Orange"},
    # Yellow sticky notes
    {"lower": (20, 80, 100), "upper": (40, 255, 255), "name": "Yellow"},
    # Green sticky notes
    {"lower": (35, 80, 100), "upper": (85, 255, 255), "name": "Green"},
    # Blue sticky notes
    {"lower": (90, 80, 80), "upper": (130, 255, 255), "name": "Blue"},
    # Pink/Magenta sticky notes
    {"lower": (140, 60, 100), "upper": (175, 255, 255), "name": "Pink"},
    # Red (wraps around 0)
    {"lower": (0, 100, 100), "upper": (5, 255, 255), "name": "Red-lo"},
    {"lower": (175, 100, 100), "upper": (180, 255, 255), "name": "Red-hi"},
]

def detect_colored_rects(frame, min_area=1500, max_area=120000):
    """Detect bright-colored rectangular sticky notes using HSV color ranges.
    Much more stable than edge-based methods."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected = []

    # Build combined mask of ALL target colors
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for cr in COLOR_RANGES:
        mask = cv2.inRange(hsv, cr["lower"], cr["upper"])
        full_mask = cv2.bitwise_or(full_mask, mask)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Solidity check
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        if area / hull_area < 0.6:
            continue

        # Fit a rotated rectangle (always gives 4 clean points)
        rect = cv2.minAreaRect(cnt)
        ww, hh = rect[1]
        if ww == 0 or hh == 0:
            continue
        aspect = max(ww, hh) / min(ww, hh)
        if aspect > 4:
            continue

        box = cv2.boxPoints(rect)
        pts = np.int32(box)

        # Deduplicate
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        dup = False
        for ep in detected:
            ecx = int(np.mean(ep[:, 0]))
            ecy = int(np.mean(ep[:, 1]))
            if math.hypot(cx - ecx, cy - ecy) < 60:
                dup = True
                break
        if not dup:
            detected.append(pts)

    return detected


# ─────────────────────────────────────────────
# Temporal Stabilizer (prevents flickering)
# ─────────────────────────────────────────────
class ObstacleStabilizer:
    """Requires consistent detection before showing an obstacle,
    and keeps it alive briefly after it disappears."""
    def __init__(self, confirm_frames=4, hold_frames=10, match_dist=80):
        self.confirm_frames = confirm_frames
        self.hold_frames = hold_frames
        self.match_dist = match_dist
        self.tracked = []  # list of {"pts", "center", "seen", "absent", "confirmed"}

    def update(self, raw_detections):
        # Match raw detections to existing tracked objects
        matched_raw = set()
        matched_tracked = set()

        for i, det in enumerate(raw_detections):
            dcx = int(np.mean(det[:, 0]))
            dcy = int(np.mean(det[:, 1]))
            best_j = -1
            best_dist = self.match_dist

            for j, trk in enumerate(self.tracked):
                if j in matched_tracked:
                    continue
                d = math.hypot(dcx - trk["center"][0], dcy - trk["center"][1])
                if d < best_dist:
                    best_dist = d
                    best_j = j

            if best_j >= 0:
                # Update existing tracked obstacle with smoothed position
                trk = self.tracked[best_j]
                trk["pts"] = (0.4 * det + 0.6 * trk["pts"]).astype(np.int32)
                trk["center"] = (int(np.mean(trk["pts"][:, 0])), int(np.mean(trk["pts"][:, 1])))
                trk["seen"] += 1
                trk["absent"] = 0
                if trk["seen"] >= self.confirm_frames:
                    trk["confirmed"] = True
                matched_raw.add(i)
                matched_tracked.add(best_j)
            else:
                matched_raw.add(i)

        # New detections (unmatched raw)
        for i, det in enumerate(raw_detections):
            if i not in matched_raw or (i in matched_raw and all(
                math.hypot(int(np.mean(det[:,0])) - t["center"][0],
                           int(np.mean(det[:,1])) - t["center"][1]) > self.match_dist
                for t in self.tracked
            )):
                dcx = int(np.mean(det[:, 0]))
                dcy = int(np.mean(det[:, 1]))
                # Only add if truly new
                already = any(math.hypot(dcx - t["center"][0], dcy - t["center"][1]) < self.match_dist
                              for t in self.tracked)
                if not already:
                    self.tracked.append({
                        "pts": det.copy(),
                        "center": (dcx, dcy),
                        "seen": 1,
                        "absent": 0,
                        "confirmed": False
                    })

        # Tick absence for unmatched tracked objects
        for j, trk in enumerate(self.tracked):
            if j not in matched_tracked:
                trk["absent"] += 1
                trk["seen"] = max(0, trk["seen"] - 1)

        # Remove stale
        self.tracked = [t for t in self.tracked if t["absent"] < self.hold_frames]

    def get_confirmed(self):
        return [t["pts"] for t in self.tracked if t["confirmed"]]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    cam_idx = 0
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    balls = []
    spawn_timer = 0
    auto_spawn = True
    stabilizer = ObstacleStabilizer(confirm_frames=4, hold_frames=12)

    print("=" * 50)
    print("  AR Sticky Note Physics Sandbox v2")
    print("=" * 50)
    print("Use BRIGHT colored sticky notes (orange, yellow,")
    print("green, blue, pink) on a wall/desk.")
    print()
    print("Controls:")
    print("  SPACE  - Drop a ball")
    print("  A      - Toggle auto-spawn")
    print("  C      - Clear all balls")
    print("  N      - Switch camera (0→1→2)")
    print("  Q      - Quit")
    print("=" * 50)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Detect rectangles
        raw_rects = detect_colored_rects(frame)
        stabilizer.update(raw_rects)
        rects = stabilizer.get_confirmed()

        # Dark display
        display = cv2.convertScaleAbs(frame, alpha=0.4, beta=5)
        overlay = np.zeros_like(display, dtype=np.uint8)

        # Draw confirmed obstacles
        for pts in rects:
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 3, cv2.LINE_AA)
            fill_overlay = np.zeros_like(overlay)
            cv2.fillConvexPoly(fill_overlay, pts, (0, 60, 60))
            overlay = cv2.add(overlay, fill_overlay)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(overlay, "OBSTACLE", (cx - 35, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Auto-spawn balls (with cap)
        spawn_timer += 1
        if auto_spawn and spawn_timer >= 50 and len(balls) < MAX_BALLS:
            spawn_timer = 0
            bx = random.randint(100, w - 100)
            balls.append(Ball(bx, -10))

        # Update balls
        alive_balls = []
        for ball in balls:
            ball.update(gravity=0.5)

            # Bounce off walls only (left/right)
            if ball.x - ball.radius < 0:
                ball.x = ball.radius
                ball.vx = abs(ball.vx) * ball.restitution
            elif ball.x + ball.radius > w:
                ball.x = w - ball.radius
                ball.vx = -abs(ball.vx) * ball.restitution

            # NO floor bounce — balls fall into the void

            # Collide with rectangle edges
            for pts in rects:
                for i in range(4):
                    ax, ay = pts[i]
                    bx_s, by_s = pts[(i + 1) % 4]
                    collide_ball_segment(ball, float(ax), float(ay), float(bx_s), float(by_s))

            ball.draw(overlay)

            if ball.alive(w, h):
                alive_balls.append(ball)
        balls = alive_balls

        # Composite
        display = cv2.addWeighted(overlay, 0.9, display, 1.0, 0)

        # HUD
        cv2.rectangle(display, (0, 0), (w, 45), (15, 15, 20), -1)
        cv2.putText(display, "AR STICKY NOTE PHYSICS", (30, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
        auto_label = "ON" if auto_spawn else "OFF"
        cv2.putText(display, f"Balls: {len(balls)}/{MAX_BALLS} | Obstacles: {len(rects)} | Auto: {auto_label} | Cam: {cam_idx}",
                    (450, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("AR Sticky Note Physics", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' ') and len(balls) < MAX_BALLS:
            bx = random.randint(100, w - 100)
            balls.append(Ball(bx, 10))
        elif key == ord('a') or key == ord('A'):
            auto_spawn = not auto_spawn
        elif key == ord('c') or key == ord('C'):
            balls.clear()
        elif key == ord('n') or key == ord('N'):
            cap.release()
            cam_idx = (cam_idx + 1) % 3
            cap = cv2.VideoCapture(cam_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            stabilizer = ObstacleStabilizer(confirm_frames=4, hold_frames=12)
            print(f"Switched to camera {cam_idx}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
