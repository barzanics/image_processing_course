"""Create every example image used in the course (no downloads needed).

Run it once:   python make_assets.py
All pictures are drawn with OpenCV + NumPy and saved as lossless PNG files.
"""
import os
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")                               # draw without opening a window
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))   # folder this script lives in
ASSETS = os.path.join(HERE, "assets")               # where pictures are written
FONT = cv2.FONT_HERSHEY_SIMPLEX                     # the one font used everywhere


def save(name, img):
    """Write an image into assets/ and report it."""
    cv2.imwrite(os.path.join(ASSETS, name), img)    # PNG = lossless
    print("created assets/" + name, img.shape)


def white_canvas(height, width):
    """A white BGR picture."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def text_picture(text, scale, thickness, pad):
    """Black text on white with generous white padding around it."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    img = white_canvas(th + base + 2 * pad, tw + 2 * pad)
    cv2.putText(img, text, (pad, pad + th), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def add_noise(img, sigma):
    """Add gaussian noise to a picture (seed is set by the caller)."""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def make_letters():
    clean = text_picture("HELLO OPENCV", 2.0, 3, 60)
    save("letters_clean.png", clean)

    np.random.seed(1)                                   # same noise every run
    noisy = add_noise(clean, 40)                        # gaussian noise
    noisy = cv2.GaussianBlur(noisy, (3, 3), 0)          # slight blur
    height, width = noisy.shape[:2]
    for i in range(10):                                 # a few dark dust specks
        x = np.random.randint(10, width - 10)
        y = np.random.randint(10, height - 10)
        cv2.circle(noisy, (x, y), 3, (0, 0, 0), -1, cv2.LINE_AA)
    save("letters_noisy.png", noisy)

    height, width = clean.shape[:2]                     # skew = rotate by 10 degrees
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 10, 1.0)
    skewed = cv2.warpAffine(clean, matrix, (width, height),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    save("letters_skewed.png", skewed)


def make_digits():
    pad, scale, thick, step = 50, 2.5, 4, 85
    img = white_canvas(150, 2 * pad + 9 * step + 50)
    for d in range(10):                                 # digits 0..9, evenly spaced
        cv2.putText(img, str(d), (pad + d * step, 100), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    save("digits_single.png", img)

    img = white_canvas(190, 800)                        # two numbers on one line
    cv2.putText(img, "427", (60, 120), FONT, 3.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, "1058", (420, 120), FONT, 3.0, (0, 0, 0), 5, cv2.LINE_AA)
    save("digits_number.png", img)


def make_photo_like():
    height, width = 600, 900
    ys, xs = np.mgrid[0:height, 0:width]                # coordinate grids
    gradient = 175 + 40 * (xs / width)                  # lighter on the right
    dx = (xs - width / 2) / (width / 2)
    dy = (ys - height / 2) / (height / 2)
    vignette = 1.0 - 0.30 * (dx ** 2 + dy ** 2)         # darker in the corners
    gray = gradient * vignette
    img = np.zeros((height, width, 3), dtype=np.float64)
    img[:, :, 0] = gray * 0.95                          # tiny warm tint (B,G,R)
    img[:, :, 1] = gray * 1.00
    img[:, :, 2] = gray * 1.03
    img = np.clip(img, 0, 255).astype(np.uint8)
    # (text, x, y of baseline, scale, thickness)
    numbers = [("52", 60, 130, 2.6, 5), ("7319", 430, 120, 2.0, 4),
               ("908", 170, 320, 3.0, 5), ("2025", 70, 500, 2.2, 4),
               ("46", 520, 490, 1.8, 3)]
    for text, x, y, scale, thick in numbers:
        cv2.putText(img, text, (x, y), FONT, scale, (45, 45, 45), thick, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)              # camera softness
    np.random.seed(7)
    img = add_noise(img, 3)                             # a little sensor noise
    save("digits_grid_photo_like.png", img)


def draw_triangle(img, x, y, w, h, color):
    points = np.array([[x, y + h], [x + w // 2, y], [x + w, y + h]], dtype=np.int32)
    cv2.fillPoly(img, [points], color, cv2.LINE_AA)


def make_objects():
    img = white_canvas(420, 600)
    red = [(0, 0, 220), (30, 20, 200), (10, 10, 240)]
    green = [(0, 170, 0), (40, 150, 20), (20, 190, 30)]
    blue = [(230, 60, 0), (200, 40, 20), (255, 80, 10)]
    circles = [(100, 90, 40), (260, 95, 28), (420, 85, 48)]
    for i in range(3):                                  # red circles
        x, y, r = circles[i]
        cv2.circle(img, (x, y), r, red[i], -1, cv2.LINE_AA)
    squares = [(70, 175, 70), (250, 185, 50), (390, 170, 80)]
    for i in range(3):                                  # green squares
        x, y, s = squares[i]
        cv2.rectangle(img, (x, y), (x + s, y + s), green[i], -1)
    triangles = [(70, 300, 80, 70), (240, 310, 60, 60), (390, 295, 90, 85)]
    for i in range(3):                                  # blue triangles
        x, y, w, h = triangles[i]
        draw_triangle(img, x, y, w, h, blue[i])
    save("objects_shapes.png", img)

    scene = white_canvas(600, 800)                      # the messy scene
    cv2.circle(scene, (130, 130), 70, (40, 120, 30), -1, cv2.LINE_AA)    # big circle
    cv2.circle(scene, (330, 110), 45, (30, 30, 170), -1, cv2.LINE_AA)    # medium circle
    cv2.circle(scene, (520, 100), 28, (160, 70, 20), -1, cv2.LINE_AA)    # small circle
    cv2.circle(scene, (680, 110), 38, (30, 30, 170), -1, cv2.LINE_AA)    # two circles
    cv2.circle(scene, (725, 140), 38, (30, 30, 170), -1, cv2.LINE_AA)    # that overlap
    cv2.rectangle(scene, (250, 230), (340, 320), (160, 70, 20), -1)      # big square
    cv2.rectangle(scene, (470, 250), (525, 305), (40, 120, 30), -1)      # small square
    cv2.rectangle(scene, (40, 380), (260, 400), (60, 60, 60), -1)        # long flat bar
    cv2.rectangle(scene, (600, 230), (618, 400), (60, 60, 60), -1)       # tall thin bar
    draw_triangle(scene, 60, 450, 80, 70, (30, 30, 170))                 # triangle #1
    draw_triangle(scene, 400, 460, 80, 70, (30, 30, 170))                # same triangle #2
    draw_triangle(scene, 640, 450, 110, 95, (160, 70, 20))               # bigger triangle
    np.random.seed(3)                                   # dust specks: same every run
    placed = 0
    while placed < 10:
        x = np.random.randint(15, 785)
        y = np.random.randint(15, 585)
        patch = scene[y - 12:y + 13, x - 12:x + 13]     # only place on empty background
        if patch.min() == 255:
            cv2.circle(scene, (x, y), 2, (50, 50, 50), -1, cv2.LINE_AA)
            placed = placed + 1
    save("objects_scene.png", scene)


def make_real_font():
    """Digits in a 'real-world' font (DejaVu Sans Mono, shipped with matplotlib)."""
    fig = plt.figure(figsize=(7, 4), dpi=100)           # 700 x 400 pixels
    fig.patch.set_facecolor("#e6ecf0")                  # soft blue-gray paper
    words = [("2026", 0.10, 0.62, 44), ("88", 0.62, 0.62, 36), ("9", 0.10, 0.15, 52),
             ("305", 0.40, 0.15, 44), ("1999", 0.74, 0.15, 30)]
    for text, x, y, size in words:
        fig.text(x, y, text, fontsize=size, fontfamily="DejaVu Sans Mono",
                 fontweight="bold", color="#1c2b4a")
    fig.canvas.draw()                                   # render to memory
    rgba = np.asarray(fig.canvas.buffer_rgba())         # pixels as an array
    img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)        # OpenCV wants BGR
    plt.close(fig)
    save("digits_real_font.png", img)


def make_incoming_sample():
    """A stand-in for the teacher's photo. The teacher's own file is myimage.png."""
    img = white_canvas(360, 700)
    img[:] = (235, 240, 240)                            # off-white "paper"
    cv2.putText(img, "1375", (330, 95), FONT, 2.2, (40, 40, 40), 4, cv2.LINE_AA)
    cv2.putText(img, "86", (60, 250), FONT, 3.0, (40, 40, 40), 5, cv2.LINE_AA)
    cv2.putText(img, "9041", (330, 280), FONT, 1.8, (40, 40, 40), 3, cv2.LINE_AA)
    cv2.imwrite(os.path.join(ASSETS, "incoming", "sample_numbers.png"), img)
    print("created assets/incoming/sample_numbers.png", img.shape)
    note = os.path.join(ASSETS, "incoming", "README.txt")
    if not os.path.exists(note):                        # leave instructions for the teacher
        with open(note, "w") as f:
            f.write("Drop your own photos here (for example myimage.png).\n"
                    "Then change the IMAGE_PATH line at the top of a notebook.\n"
                    "sample_numbers.png is a ready-made practice image.\n")


if __name__ == "__main__":
    for folder in ["", "incoming", "out", "digit_templates"]:
        os.makedirs(os.path.join(ASSETS, folder), exist_ok=True)
    make_letters()
    make_digits()
    make_photo_like()
    make_objects()
    make_real_font()
    make_incoming_sample()
    print("All assets are ready.")
