import sys
print(sys.executable)
import cv2
import numpy as np
import os

os.makedirs("out", exist_ok=True)

# ---------- 1. Build a synthetic test image ----------
img = np.full((500, 800, 3), 255, dtype=np.uint8)

cv2.circle   (img, (120, 120), 60, (0, 0, 255), -1)              # red circle
cv2.rectangle(img, (220,  60), (320, 160), (0, 0, 255), -1)      # red square
cv2.rectangle(img, (350, 100), (550, 130), (0, 0, 255), -1)      # red long bar (should be filtered)
cv2.rectangle(img, (100, 250), (200, 350), (255, 0, 0), -1)      # blue
pts = np.array([[400, 200], [350, 300], [450, 300]], np.int32)
cv2.fillPoly(img, [pts], (0, 255, 0))                            # green triangle
cv2.rectangle(img, (600, 250), (640, 290), (0, 255, 255), -1)    # yellow small
cv2.circle   (img, (700, 400), 3, (0, 0, 255), -1)               # tiny red noise dot

cv2.imwrite("out/07_input.png", img)

# ---------- 2. HSV + red mask (two ranges) ----------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low_red1,  high_red1  = np.array([0,   100, 100]), np.array([10,  255, 255])
low_red2,  high_red2  = np.array([170, 100, 100]), np.array([179, 255, 255])

mask_a = cv2.inRange(hsv, low_red1, high_red1)
mask_b = cv2.inRange(hsv, low_red2, high_red2)
red_mask = cv2.bitwise_or(mask_a, mask_b)
cv2.imwrite("out/07_red_mask.png", red_mask)

only_red = cv2.bitwise_and(img, img, mask=red_mask)
cv2.imwrite("out/07_only_red.png", only_red)

# ---------- 3. Contours + area filter ----------
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"[07] Raw red contours: {len(contours)}")

boxed = img.copy()
kept_boxes = []
for cnt in contours:
    if cv2.contourArea(cnt) >= 200:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (255, 0, 255), 2)
        kept_boxes.append((x, y, w, h))
cv2.imwrite("out/07_red_boxed.png", boxed)
print(f"[07] After area >= 200: {len(kept_boxes)}")

# ---------- 4. Shape analysis: area, aspect ratio, circularity ----------
print("\n[07] Shape analysis (red objects):")
compact = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 200:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    ratio       = w / h
    perimeter   = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    tag = "compact" if 0.5 <= ratio <= 2.0 else "elongated"
    print(f"   box=({x:3},{y:3},{w:3},{h:3})  area={area:6.0f}  "
          f"ratio={ratio:5.2f}  circ={circularity:4.2f}  -> {tag}")
    if 0.5 <= ratio <= 2.0:
        compact.append((x, y, w, h))

print(f"[07] Compact shapes: {len(compact)}")

# ---------- 5. Template matching + NMS ----------
print("\n[07] Template matching demo")
scene = np.full((400, 600, 3), 255, dtype=np.uint8)

def draw_triangle(img, cx, cy, size, color):
    pts = np.array([[cx, cy - size],
                    [cx - size, cy + size // 2],
                    [cx + size, cy + size // 2]], np.int32)
    cv2.fillPoly(img, [pts], color)

draw_triangle(scene, 100, 100, 40, (0, 0, 255))
draw_triangle(scene, 300, 150, 40, (0, 0, 255))
draw_triangle(scene, 480, 250, 40, (0, 0, 255))
cv2.imwrite("out/07_scene.png", scene)

template = scene[55:145, 55:145].copy()
cv2.imwrite("out/07_template.png", template)

g_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
g_tmpl  = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(g_scene, g_tmpl, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)
print(f"[07] Best match value={max_val:.3f} at {max_loc}")

ys, xs = np.where(result >= 0.95)
print(f"[07] Raw hits (>=0.95): {len(xs)}")

th, tw = g_tmpl.shape
hits = sorted(zip(result[ys, xs].tolist(), xs.tolist(), ys.tolist()), reverse=True)
kept = []
for score, x, y in hits:
    overlap = any(abs(x - kx) < tw and abs(y - ky) < th for kx, ky in kept)
    if not overlap:
        kept.append((x, y))
print(f"[07] After NMS: {len(kept)} unique detections")

found = scene.copy()
for x, y in kept:
    cv2.rectangle(found, (x, y), (x + tw, y + th), (0, 255, 0), 2)
cv2.imwrite("out/07_template_found.png", found)

print("\n[07] DONE — inspect files under out/07_*.png")