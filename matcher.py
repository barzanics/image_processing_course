"""The finished character reader from notebooks 05, 06 and 08.

Notebooks 05, 06 and 08 build each of these functions step by step.
Notebook 09 and the exercise solutions simply re-use the finished versions.
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX           # the font used to draw templates
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # letters we can recognize
DIGITS = "0123456789"                     # digits we can recognize


def tidy(crop):
    """Squash any cropped character into a 32 x 32 picture."""
    return cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)


def make_templates(characters):
    """Draw each character with cv2.putText and return {character: 32x32 picture}."""
    templates = {}
    for character in characters:
        canvas = np.zeros((120, 120), dtype=np.uint8)                       # black canvas
        cv2.putText(canvas, character, (20, 90), FONT, 2.5, 255, 4, cv2.LINE_AA)
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)       # pure black/white
        x, y, w, h = cv2.boundingRect(canvas)                                # tight box
        templates[character] = tidy(canvas[y:y + h, x:x + w])
    return templates


def recognize(crop, templates):
    """Return (best character, its score, runner-up character, runner-up score)."""
    small = tidy(crop)
    best, best_score, second, second_score = "?", -2.0, "?", -2.0
    for character in templates:
        score = cv2.matchTemplate(small, templates[character], cv2.TM_CCOEFF_NORMED)[0][0]
        if score > best_score:                                  # new champion
            second, second_score = best, best_score
            best, best_score = character, score
        elif score > second_score:                              # new runner-up
            second, second_score = character, score
    return best, float(best_score), second, float(second_score)


def find_boxes(gray, use_median=False, min_area=10):
    """gray -> blur -> Otsu -> invert -> contours -> area filter -> boxes sorted left to right."""
    if use_median:
        blur = cv2.medianBlur(gray, 5)
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)                            # text becomes white (255)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:               # ignore tiny specks
            boxes.append(cv2.boundingRect(contour))
    return thresh, sorted(boxes)


def find_boxes_adaptive(gray, min_area=80):
    """For uneven lighting: gray -> blur -> adaptive threshold -> close -> contours -> boxes."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 10)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            boxes.append(cv2.boundingRect(contour))
    return closed, boxes


def group_rows(boxes, row_gap=40):
    """Group boxes into rows by their vertical centre; each row is sorted left to right."""
    centres = []
    for x, y, w, h in boxes:
        centres.append((y + h / 2, x, y, w, h))                # centre-y first, so sorting works
    centres.sort()                                             # top to bottom
    rows, current, last_y = [], [], None
    for cy, x, y, w, h in centres:
        if last_y is not None and cy - last_y > row_gap:       # big jump = new row
            rows.append(sorted(current))
            current = []
        current.append((x, y, w, h))
        last_y = cy
    if len(current) > 0:
        rows.append(sorted(current))
    return rows


def read_row(thresh, row, templates):
    """Read one row of boxes. Returns a list of number strings (big gaps split numbers)."""
    numbers, text, last_right = [], "", None
    for x, y, w, h in row:
        if last_right is not None and x - last_right > 0.8 * h:    # big gap = new number
            numbers.append(text)
            text = ""
        text = text + recognize(thresh[y:y + h, x:x + w], templates)[0]
        last_right = x + w
    numbers.append(text)
    return numbers


def read_all_numbers(gray):
    """The whole capstone in one call: returns a list of rows, each a list of number strings."""
    closed, boxes = find_boxes_adaptive(gray)
    templates = make_templates(DIGITS)
    result = []
    for row in group_rows(boxes):
        result.append(read_row(closed, row, templates))
    return result
