"""
تشخیص یک حرف انگلیسی از روی تصویر.
استفاده: تصویر خودت رو در مسیر IMAGE_PATH بذار و اجرا کن.
"""
import sys
import cv2
import numpy as np

# ---------- 1. تنظیمات ----------
IMAGE_PATH = "assets/letters_noisy.png"          # تصویر ورودی (سیاه روی سفید یا سفید روی سیاه)
MIN_AREA = 10                      # حداقل مساحت لکه (برای حذف نویز)
SIZE = 32                          # اندازه‌ی استاندارد قالب‌ها
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ---------- 2. تابع tidy ----------
def tidy(crop):
    """هر تصویری رو به SIZE×SIZE تبدیل می‌کنه."""
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

# ---------- 3. ساخت قالب‌ها ----------
def build_templates():
    """یک دیکشنری از A تا Z، هر کدوم یک تصویر SIZE×SIZE سفید روی سیاه."""
    tpl = {}
    for letter in ALPHABET:
        canvas = np.zeros((120, 120), dtype=np.uint8)
        cv2.putText(canvas, letter, (20, 90), FONT, 2.5, 255, 4, cv2.LINE_AA)
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(canvas)
        tpl[letter] = tidy(canvas[y:y+h, x:x+w])
    return tpl

# ---------- 4. پیدا کردن جعبه‌ی حرف ----------
def find_letter_box(gray, min_area=10, use_median=False):
    """
    خروجی: (thresh, boxes)
    thresh = تصویر سفید-روی-سیاه
    boxes = لیست تاپل‌های (x,y,w,h) مرتب‌شده از چپ به راست
    """
    if use_median:
        blur = cv2.medianBlur(gray, 5)
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(otsu)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            boxes.append(cv2.boundingRect(c))
    boxes.sort()
    return thresh, boxes

# ---------- 5. تشخیص یک حرف ----------
def recognize(crop, templates):
    """
    ورودی: تصویر یک حرف (سفید روی سیاه)
    خروجی: (best_letter, best_score, all_scores_dict)
    """
    small = tidy(crop)
    all_scores = {}
    best_letter = "?"
    best_score = -2.0

    for letter, tpl in templates.items():
        score = float(cv2.matchTemplate(small, tpl, cv2.TM_CCOEFF_NORMED)[0][0])
        all_scores[letter] = score
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter, best_score, all_scores

# ---------- 6. اجرای اصلی ----------
def main():
    # 6-1. خوندن تصویر
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"خطا: تصویر '{IMAGE_PATH}' پیدا نشد.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 6-2. پیدا کردن جعبه‌ها
    thresh, boxes = find_letter_box(gray, MIN_AREA)
    print("تعداد لکه‌های پیدا شده:", len(boxes))

    if len(boxes) == 0:
        print("هیچ لکه‌ای پیدا نشد. MIN_AREA رو کمتر کن یا تصویر رو چک کن.")
        return

    # 6-3. ساخت قالب‌ها
    templates = build_templates()

    # 6-4. برای هر لکه تشخیص بده (معمولاً فقط یک لکه داریم)
    for i, (x, y, w, h) in enumerate(boxes):
        crop = thresh[y:y+h, x:x+w]
        letter, score, all_scores = recognize(crop, templates)

        # مرتب کردن نتایج برای نمایش top-3
        ranked = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)
        runner_up, runner_score = ranked[1]

        margin = score - runner_score

        print(f"\n--- لکه‌ی {i+1} ---")
        print(f"  مکان: x={x}, y={y}, w={w}, h={h}")
        print(f"  تشخیص: {letter}")
        print(f"  اطمینان: {score:.2f}")
        print(f"  نفر دوم: {runner_up} با {runner_score:.2f}")
        print(f"  margin: {margin:.2f}")
        print(f"  سه نتیجه‌ی برتر:")
        for L, s in ranked[:3]:
            print(f"    {L}: {s:.3f}")

        # هشدار
        if score < 0.7:
            print("  ⚠ اطمینان پایین! احتمال خطا زیاد است.")
        elif margin < 0.1:
            print("  ⚠ تصمیم نزدیک! حرف شبیه دیگری است.")

if __name__ == "__main__":
    main()