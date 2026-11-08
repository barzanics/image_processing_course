"""Tiny helper functions used by every notebook.

Nothing fancy here: each function is a few lines so students can read it.
"""
import cv2                       # OpenCV: reading and processing images
import matplotlib.pyplot as plt  # matplotlib: showing images on screen


def load(path):
    """Read an image file from disk and return it as a NumPy array (BGR)."""
    img = cv2.imread(path)  # cv2.imread returns None if the file is missing
    if img is None:
        # Stop with a friendly message instead of a confusing error later on
        raise FileNotFoundError(
            "Could not read '" + path + "'. Did you run  python make_assets.py  first?"
        )
    return img


def bgr2rgb(img):
    """Swap OpenCV's Blue-Green-Red order into matplotlib's Red-Green-Blue order."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show(img, title="", size=(6, 4)):
    """Show ONE image (color or gray) with matplotlib."""
    plt.figure(figsize=size)                      # make a blank figure of a given size
    if len(img.shape) == 2:                       # 2 numbers in shape -> gray image
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)  # fixed 0-255 so gray is honest
    else:                                         # 3 numbers in shape -> color image
        plt.imshow(bgr2rgb(img))                  # fix the BGR/RGB order first
    plt.title(title)                              # write the title above the picture
    plt.axis("off")                               # hide the axis numbers
    plt.show()                                    # draw it in the notebook


def show_many(images, titles, columns=4, size=None):
    """Show several images side by side in a grid, each with its own title."""
    count = len(images)                           # how many pictures do we have?
    rows = (count + columns - 1) // columns       # rows needed (round up)
    if size is None:                              # choose a sensible figure size
        size = (3 * columns, 3 * rows)
    plt.figure(figsize=size)                      # one big figure holds the grid
    for i in range(count):                        # go through the pictures one by one
        plt.subplot(rows, columns, i + 1)         # pick the i-th cell of the grid
        img = images[i]                           # take the i-th picture
        if len(img.shape) == 2:                   # gray picture?
            plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:                                     # color picture
            plt.imshow(bgr2rgb(img))
        plt.title(titles[i], fontsize=9)          # small title for this picture
        plt.axis("off")                           # hide axis numbers
    plt.tight_layout()                            # stop titles overlapping
    plt.show()                                    # draw the whole grid
