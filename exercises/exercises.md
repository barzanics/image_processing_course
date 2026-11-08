# Exercises

Each task is phrased as **"given this image, produce this result"**. Solved notebooks are in `exercises/solutions/` (one per set). Pictures live in `assets/` (run `python make_assets.py` first). Numbers in *italics* are the expected output.

---
## Set 01 - Images are numbers  (`solutions_01_images_are_numbers.ipynb`)
1. **Given** `letters_clean.png`, **print** `The picture is W pixels wide and H pixels tall.` -> *553 wide, 175 tall*.
2. **Given** `objects_shapes.png`, **print** the B, G, R values at row 350, column 110 and name the colour. -> *`[230 60 0]`, blue*.
3. **Given** `objects_shapes.png`, **crop** rows 185-235, columns 250-300 (the middle green square), show it and print its shape. -> *`(50, 50, 3)`*.
4. **Given** `objects_shapes.png`, **paint** a 20 x 20 red square in the top-left corner of a *copy*. -> *pixel (5, 5) is `[0 0 255]`, the original is still `[255 255 255]`*.

## Set 02 - Colour and grayscale  (`solutions_02_...`)
1. **Given** `objects_shapes.png` in gray, which is brighter: the first red circle (row 90, col 100) or the first green square (row 210, col 105)? -> *66 vs 100: the green square*.
2. **Show** the blue channel as a gray picture and print its value inside a blue triangle (row 350, col 110) and inside a red circle (row 90, col 100). -> *230 and 0*.
3. **Make** the picture 50% darker with `cv2.convertScaleAbs` and print its brightest value. -> *128*.
4. **Make** a thumbnail 100 pixels wide that keeps the proportions. -> *shape `(70, 100, 3)`*.

## Set 03 - Thresholding and cleaning  (`solutions_03_...`)
1. **Apply** Otsu to `letters_clean.png`; print the threshold chosen. -> *131.0*.
2. **Flip** the result to white-on-black and count white pixels. -> *8403*.
3. **Clean** `letters_noisy.png` with median blur (5) + Otsu and show before and after. -> *only values 0 and 255 remain*.

## Set 04 - Contours and bounding boxes  (`solutions_04_...`)
1. **Count** the contours in `digits_single.png` and print each box width. -> *10 contours, widths 37-44*.
2. **Count** the contours with area below 60 in `letters_noisy.png` (median blur + Otsu first). -> *18 contours: 7 specks, 11 letters*.
3. **Print** the left edge (x) of every digit in `digits_number.png`, left to right. -> *`[62, 120, 178, 426, 480, 536, 591]`*.

## Set 05 - Reading letters  (`solutions_05_...`)
1. **Read** `letters_clean.png` and print the weakest letter confidence. -> *about 0.96*.
2. **Draw** the word `OPEN` with `cv2.putText` and read it back. -> *`OPEN`*.
3. **Deskew** `letters_skewed.png` by rotating -10 degrees, then read it. -> *`HELLO OPENCV`*.

## Set 06 - Reading digits to values  (`solutions_06_...`)
1. **Read** `digits_single.png`; are all 10 right? -> *`0123456789`, True*.
2. **Read** `digits_number.png`; print both numbers and their sum. -> *427, 1058, sum 1485*.
3. **Draw** `90210` yourself and read it as an `int`. -> *`90210`, type `int`*.

## Set 07 - Detect and extract objects  (`solutions_07_...`)
1. **Find** the area of the biggest green object in `objects_shapes.png`. -> *6400.0*.
2. **Count** circles with area above 5000 in `objects_scene.png` (use circularity). -> *2*.
3. **Cut** the big square (with a 10 px margin: rows 220-330, columns 240-350) out of the scene and find it again with `cv2.matchTemplate`. -> *score 1.0 at `(240, 220)`*.

## Set 08 - Capstone  (`solutions_08_...`)
1. **Print** only the 4-digit numbers in `digits_grid_photo_like.png`. -> *`[7319, 2025]`*.
2. **Add up** every number in the picture. -> *10350*.
3. **Save** a crop of the biggest digit to `assets/out/biggest_digit.png`. -> *box `(286, 258, 49, 63)`, file saved*.

---
## Final mini-project: "extract every number"
**Given** `assets/incoming/myimage.png` (a photo the teacher supplies), **extract and print every number in it**, row by row, and draw a green box around every digit.

The solution `solutions_final_mini_project.ipynb` works on `assets/incoming/sample_numbers.png`, a ready-made practice picture, so students can rehearse first. Expected output for the sample: *`[1375, 86, 9041]`*. To use the teacher's picture, change the single line `IMAGE_PATH`.

**Tip for teachers:** the reader works best on dark digits, a plain font, upright text and reasonably even light. For a hard photo, show students *why* it fails (Notebook 08, "where this would fail") and then Notebook 09.
