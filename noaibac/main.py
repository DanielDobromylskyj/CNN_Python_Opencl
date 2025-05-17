import os
from PIL import Image
from scipy.ndimage import label
import numpy as np

OPENSLIDE_PATH = r'\openslide-win64\bin'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(os.getcwd() + OPENSLIDE_PATH):
        import openslide
else:
    import openslide

with open("../pycharm_project/trainingPoints.txt", "r") as f:
    trainingData = eval(f.read())


def view(arr):
    arr = arr * 255
    arr_uint8 = arr.astype(np.uint8)

    img = Image.fromarray(arr_uint8)
    img.show()

def scan_region(slide: openslide.OpenSlide, xy):
    region = slide.read_region(xy, 0, (100, 100))
    numpy_region = np.array(region.convert("RGB")) / 255

    inverted = 1 - numpy_region
    green_channel = inverted[:, :, 1]

    blocked = green_channel.reshape(50, 2, 50, 2).mean(axis=(1, 3))
    filtered = np.where(blocked < 0.2, 0, blocked)

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled_array, num_features = label(filtered > 0, structure=structure)

    sizes = np.bincount(labeled_array.ravel())[1:]

    for size in sizes:
        if 5 < size < 12:
            return True

    return False


if __name__ == "__main__":
    total_scanned = 0
    corrected_scanned = 0

    false_positive = 0
    false_negative = 0

    for file in trainingData:
        slide = openslide.open_slide(f"../pycharm_project/{file['tiffPath']}")

        for point in file['points']:
            xy, result = point

            scan_result = int(scan_region(slide, xy))
            if scan_result == result[0]:
                corrected_scanned += 1

            else:
                if scan_result == 1:
                    false_positive += 1
                else:
                    false_negative += 1



            total_scanned += 1

    print("Accuracy (%):", round(corrected_scanned / total_scanned * 100, 2))

    print("Correct:", corrected_scanned)
    print("False +ve:", false_positive)
    print("False -ve:", false_negative)



